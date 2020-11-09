import glob
import os
import json


class JSONtoLaTeXtableParser:

    def __init__(self, output_file, header_titles):
        """
        Constructor. Set path where the JSON files are stored,
        e. g. "/some/path/to/results/*/*.json"
        Also filename for output file is needed
        """
        self.header = header_titles
        self.lines = []
        self.out = output_file


    def __build_latex_tab_string(self, data, args):
        """ extract values from data using keywords from args """
        s = ""
        for i, key in enumerate(args):
            if i < len(args) - 1:
                s += str(data[key]) + " & "
            else:
                s += str(data[key]) + "\\" + "\\ \\hdashline"
        return s

    def convert_num_format(self, str_num):
        f1 = float(str_num[str_num.find('%')+1:str_num.find('\u00b1')-1])
        f2 = float(str_num[str_num.find('\u00b1')+2:])
        res = "{:.3f} \u00b1 {:.3f}".format(f1 / 100, f2 / 100)
        return res

    def add_line(self, data, data_keys):
        s=""
        for key in data_keys:
            if type(key) == str:
                t = str(data[key])
                if t.find('%') != -1:
                    t = self.convert_num_format(t)
                
                t = t.replace('\u00b1','\\pm')
                t = t.replace('_','\_')
                s += '$' + t + "$ & "
            elif type(key) == tuple:
                new_keys=key
                while type(new_keys) == tuple:
                    data = data[new_keys[0]]
                    new_keys = new_keys[1]
                for key_d in new_keys:
                    t = str(data[key_d])
                    if t.find('%') != -1:
                        t = self.convert_num_format(t)
                    
                    t = t.replace('\u00b1','\\pm')
                    s += '$' + t + "$ & "
        i = s.rfind('&')
        s = s[:i]
        s += "\\" + "\\"
        self.lines.append(s)

    


    def parse_second_level_to_latex_tab(self, first_level_key, *args):
        """ extract data from a dictionary within a dictionary """
        self.header= args
        for file_n in self.files:
            with open(file_n, 'r') as fd:
                data = json.load(fd)[first_level_key]
                s = self.__build_latex_tab_string(data, args)
                self.lines.append(s)


 

    def create_latex_table_2(self):
        """
        create simple LaTeX table based on extracted data.
        """
        self.table = "\\begin{tabular}{@{}l "
        self.table += (len(self.header)-1) * "r " + "@{}}\\toprule \n"
        for i, h in enumerate(self.header):
            if i < len(self.header) - 1:
                self.table += "\\textbf{" + h.replace("_", " ") + "} & "
            else:
                self.table += "\\textbf{" + h.replace("_", " ") + "} \\" + "\\ \\bottomrule \n"
        for line in self.lines:
            self.table += line + "\n"
        self.table += "\\bottomrule \n"
        self.table += "\\end{tabular}"


    def write_rows_to_file(self):
        """ write only rows with extracted data to file"""
        with open(self.out, 'w') as fd:
            for line in self.lines:
                fd.write(line + "\n")


    def write_table_to_file(self):
        """ write complete table to file """
        with open(self.out, 'w') as fd:
            fd.write(self.table)

