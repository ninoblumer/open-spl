""" read xl2 data and extract measurement"""
from __future__ import annotations
from abc import ABC, abstractmethod

import pandas
from pathlib import Path

datafolder = Path("data")

class XL2ParseError(Exception):
    pass


class XL2_SLM_Measurement:
    def __init__(self, project, name, root=datafolder):
        self.root = root
        self.project = project
        self.name = name

        self.files: dict[str, XL2_SLM_File] = dict()
        self._load_all()

    def _load_all(self):
        dir = Path(self.root) / self.project
        files = list(dir.glob(f"{self.name}_*.txt"))
        for file in files:
            self.files[file.stem.lstrip(self.name+"_")] = XL2_SLM_File(file)


class XL2_SLM_File:
    def __init__(self, filepath: Path):
        # TODO: fix missing Time section for 123_Report
        self.filepath = filepath
        if not filepath.is_file():
            raise FileNotFoundError(filepath)
        with open(filepath, "r", encoding="ascii") as f:
            self.lines = f.readlines()

        s = self.lines[0].split('\t\t')

        self.type = s[0].rstrip(':')
        self.name = s[-1].strip()

        # split into sections
        self._sections_raw = {}
        start_idx = None
        section_header = None
        for i, line in enumerate(self.lines[2:], start=2):
            if line == "\n":
                continue
            if line.startswith("#"):
                if section_header is not None:
                    self._sections_raw[section_header] = (start_idx, i-1)
                section_header = line.lstrip('#').strip()
                start_idx = i+1
        if section_header is not None:
            self._sections_raw[section_header] = (start_idx, len(self.lines))

        # self.debug_contents_toc(indexing=1)

        self.sections: dict[str, _Section] = dict()
        for section_header in self._sections_raw.keys():
            self.sections[section_header] = self.parse_section(section_header)

    def debug_contents_toc(self, indexing=0):
        print(f"[ {self.name} ]")
        print(f"type: {self.type}")
        for header, (start, stop) in self._sections_raw.items():
            print(f"\t{header}: {start+indexing}-{stop+indexing}")

    def parse_section(self, section_header):
        # logic to recognise section type
        section_type = self._find_section_type(section_header)
        start, stop = self._sections_raw[section_header]
        return section_type(self, section_header)


    def _find_section_type(self, section_header) -> type:
        start, stop = self._sections_raw[section_header]
        firstline = self.lines[start]
        token = firstline.split("\t")

        if token[0].strip().endswith(":"):
            return _SectionKeyValue
        elif section_header.startswith("Broadband LOG"):
            return _SectionTable_123_Log
        elif section_header.startswith("Broadband"):
            return _SectionTable_123_Report
        elif section_header.startswith("RTA LOG"):
            return _SectionTable_RTA_Log
        elif section_header.startswith("RTA"):
            return _SectionTable_RTA_Report
        else:
            return _SectionPlainText

class _Section(ABC):
    def __init__(self, parent: XL2_SLM_File, header):
        self.header = header
        self._start, self._stop = parent._sections_raw[header]

    @abstractmethod
    def _parse(self, parent: XL2_SLM_File):
        pass

class _SectionKeyValue(_Section):
    def __init__(self, parent: XL2_SLM_File, header):
        super().__init__(parent, header)
        self.content: dict[str, str] = dict()
        self._parse(parent)

    def _parse(self, parent: XL2_SLM_File):
        lastkey = None
        for line in parent.lines[self._start:self._stop]:
            token = line.lstrip('\t').split('\t')
            token[0] = token[0].strip()
            token[1] = token[1].strip()
            if token[0] == "":
                if lastkey is None:
                    raise XL2ParseError("Malformed key-value section")
                self.content[lastkey] = self.content[lastkey].append("\n" + token[1])
            else:
                lastkey = token[0].rstrip(":")
                self.content[lastkey] = token[2]

class _SectionPlainText(_Section):
    def __init__(self, parent: XL2_SLM_File, header):
        super().__init__(parent, header)
        self.content = ""
        self._parse(parent)

    def _parse(self, parent: XL2_SLM_File):
        self.content = "\n".join(parent.lines[self._start:self._stop])

class _SectionTable(_Section):
    def __init__(self, parent: XL2_SLM_File, header):
        super().__init__(parent, header)
        self.column_headers = []
        self.df: pandas.DataFrame | None = None
        self._parse(parent)

    @staticmethod
    def _split_line(line):
        return [x.strip() for x in line.lstrip('\t').split('\t')]

    def _postprocess(self):
        if self.df is None:
            return
        self.df.replace({"-.-": None, "": None}, inplace=True)
        # for col in self.df.columns:
        #     self.df[col] = pandas.to_numeric(self.df[col], errors="ignore")

class _SectionTable_123_Log(_SectionTable):
    def _parse(self, parent: XL2_SLM_File):
        over_whole_log_period = "over whole log period"
        if self.header.endswith(over_whole_log_period):
            sibling_header = self.header.rstrip(over_whole_log_period).strip()
            sibling_section = parent.sections.get(sibling_header)
            if sibling_section is None:
                raise XL2ParseError(f"Section {sibling_header} was not parsed yet")
            self.column_headers = sibling_section.column_headers
            lines = parent.lines[self._start:self._stop]
        else:
            self.column_headers = self._split_line(parent.lines[self._start])
            lines = parent.lines[self._start+2:self._stop] # skip units row
        rows = []
        for line in lines:
            if not line.strip(): # skip empty lines
                continue
            rows.append(self._split_line(line))

        self.df = pandas.DataFrame(rows, columns=self.column_headers)
        self._postprocess()

class _SectionTable_123_Report(_SectionTable):
    def _parse(self, parent: XL2_SLM_File):
        start_stop_headers = self._split_line(parent.lines[self._start])
        self.column_headers = self._split_line(parent.lines[self._start+1])
        self.column_headers[0] = start_stop_headers[0]+" "+self.column_headers[0]
        self.column_headers[1] = start_stop_headers[0] + " " + self.column_headers[1]
        self.column_headers[2] = start_stop_headers[2] + " " + self.column_headers[2]
        self.column_headers[3] = start_stop_headers[2] + " " + self.column_headers[3]

        lines = parent.lines[self._start+3:self._stop] # skip units row
        rows = []
        for line in lines:
            if not line.strip(): # skip empty lines
                continue
            rows.append(self._split_line(line))

        self.df = pandas.DataFrame(rows, columns=self.column_headers)
        self._postprocess()


class _SectionTable_RTA_Log(_SectionTable):
    def _parse(self, parent: XL2_SLM_File):
        over_whole_log_period = "over whole log period"
        if self.header.endswith(over_whole_log_period):
            sibling_header = self.header.rstrip(over_whole_log_period).strip()
            sibling_section = parent.sections.get(sibling_header)
            if sibling_section is None:
                raise XL2ParseError(f"Section {sibling_header} was not parsed yet")
            self.column_headers = sibling_section.column_headers
            lines = parent.lines[self._start:self._stop]
        else:
            self.column_headers = list(zip(
                self._split_line(parent.lines[self._start]),
                self._split_line(parent.lines[self._start+1])
            ))
            lines = parent.lines[self._start+3:self._stop] # skip units row
        rows = []
        for line in lines:
            if not line.strip(): # skip empy lines
                continue
            rows.append(self._split_line(line))

        self.df = pandas.DataFrame(rows, columns=self.column_headers)
        self._postprocess()


class _SectionTable_RTA_Report(_SectionTable):
    def _parse(self, parent: XL2_SLM_File):
        self.column_headers = self._split_line(parent.lines[self._start])
        lines = parent.lines[self._start+2:self._stop] # skip units row
        rows = []
        for line in lines:
            if not line.strip(): # skip empy lines
                continue
            rows.append(self._split_line(line))

        self.df = pandas.DataFrame(rows, columns=self.column_headers)
        self.df.set_index(self.column_headers[0], inplace=True)
        self._postprocess()




