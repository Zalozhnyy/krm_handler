import os.path
import re
import numpy as np
import json
import locale

ELEMENTS_PATTERN = ' i r o o2 o3 n n2 no no2 e o- o2- no2-  no+  o2+  o+ n+ n2+ n+2' \
                   ' o+2 n+3 o+3 o+4 n+4 N+5  O+5 O+6 N+6 N+7 O+7 O+8 He He+ H H+ Al Al+'
# ELEMENTS_PATTERN = ''' i r o o2 o3 n n2 no no2 e o- o2 no+  o2+  o+ n+ n2+ n+2 o+2 n+3 o+3 o+4 n+4 N+5
#                         O+5 O+6 N+6 N+7 O+7 O+8 He He+ H H+ Al Al+'''
ELEMENTS_PATTERN_FIXED = '''r o n e o+ n+ o+2 n+2 o+3 n+3 o+4 n+4 O+5 N+5 O+6 N+6 O+7 N+7 O+8'''

ELEM_INDEX = dict(zip(ELEMENTS_PATTERN.split(), [i for i in range(len(ELEMENTS_PATTERN.split()))]))

NF_PATTERN = ' NF     ror*no2     ror*o2-'
NF_AFTER_PATTERN = '  I          RK(I)     VK(I)    ROK(I)     TK(I)    TEK(I)' \
                   '   XS(8,I)     EKINT     EK(I)    EEK(I)   ERAD     ERAE'

TIME_PATTERN = '   TIME       TAU      TAUK      TAUT      AMU       ANU     TAUKP     TAUTP'

SMALL_FILE_PATTERN = '   I     R       H        T        N         Ne      N+      ' \
                     'NU       DCB     SIGMA     SIGMA0      np    Q_FUNK   N+_bsiao'

CT_PATTERN = 'I        RK        VK        TK       ROK       REZ' \
             '       CT1       ET1       WTO       WT1       WT3  QT      PRST'
CT_END_PATTERN = 'NBP1  SK(NB+1)'

KG = 0
KT = 0
out_dict = {}
PROJECT_PATH: str = None


def line_to_set(s: str):
    return set(s.strip().split())


class BigFile:
    """
    Класс для поиска данных в файле W88_T7.txt.
    Сначала дважды ищем ELEMENTS_PATTERN, потом NF_PATTERN (такая структура файла)
    """

    def __init__(self):

        self._file_name = os.path.join(PROJECT_PATH, 'W88_T7.txt')
        self._pointer = 0
        self.decoding_def = locale.getpreferredencoding()
        self.__lines = []

        self._nf_pattern = line_to_set(NF_PATTERN)
        self._nf_after_pattern = line_to_set(NF_AFTER_PATTERN)
        self._time_pattern = line_to_set(TIME_PATTERN)

        try:
            with open(self._file_name, 'r', encoding='utf-8') as f:
                self.__lines = f.readlines()
        except UnicodeDecodeError:
            with open(self._file_name, 'r', encoding=self.decoding_def) as f:
                self.__lines = f.readlines()

    def check_end_of_file(self):
        return self._pointer < len(self.__lines) - 1

    def find_elements_key(self):
        tmp = []
        for line in range(self._pointer, len(self.__lines)):
            if ELEMENTS_PATTERN in self.__lines[line]:
                line += 1
                while NF_PATTERN not in self.__lines[line]:
                    if self.__lines[line] != '\n':
                        a = self.__lines[line].strip() + \
                            ' ' + \
                            self.__lines[line + 1].strip() + \
                            ' ' + \
                            self.__lines[line + 2].strip()
                        tmp.append(a)
                        line += 3
                    line += 1
                self._pointer = line
                return np.array([i.split() for i in tmp], dtype=float)

    def find_nf_key(self):
        tmp = []
        for line in range(self._pointer, len(self.__lines)):
            if self._nf_pattern.issubset(line_to_set(self.__lines[line])):

                if self._nf_after_pattern.issubset(line_to_set(self.__lines[line + 2])):
                    line += 3

                    while len(self.__lines[line].strip().split()) == len(self._nf_after_pattern):
                        tmp.append(self.__lines[line].strip())
                        line += 1

                    while not self._time_pattern.issubset(line_to_set(self.__lines[line])):
                        line += 1

                    self._pointer = line
                    tmp = [i.split() for i in tmp]
                    time = self.__lines[line + 1].strip().split()[0]

                    table = self.__validation(tmp)
                    table = np.array(table, float)

                    return table, float(time) * 1e-6

        raise EOFError

    def find_CT_key(self):
        tmp = []
        for line in range(self._pointer, len(self.__lines)):
            if CT_PATTERN in self.__lines[line]:
                line += 1
                while CT_END_PATTERN not in self.__lines[line]:
                    tmp.append(self.__lines[line].strip())
                    line += 1
                self._pointer = line
                tmp = [i.split() for i in tmp]

                table = self.__validation(tmp, 13)
                table = np.array(table, float)

                return table

    def __validation(self, data: list, n=12) -> list:
        for i in range(len(data)):
            if len(data[i]) != n:
                data[i] = self.__fix_space(data[i], n)
        return data

    def __fix_space(self, data: list, n: int) -> list:
        """Разделяем слипшиеся столбцы типа 7.8443157E+02-8.859E-17 """
        new_list = []
        for i in range(len(data)):
            e_symbols = re.findall('E.\d\d', data[i])
            if len(e_symbols) > 1:
                digits_split = re.split('E.\d\d', data[i])
                for j in range(len(digits_split)):
                    if digits_split[j] != '':
                        new_list.append(digits_split[j] + e_symbols[j])
                continue
            new_list.append(data[i])

        assert len(new_list) == n
        return new_list


class SmallFile:
    def __init__(self):
        self._file_name = os.path.join(PROJECT_PATH, r'W88_T7_RF.TXT')
        self._pointer = 0
        self.decoding_def = locale.getpreferredencoding()

        try:
            with open(self._file_name, 'r', encoding='utf-8') as f:
                self.__lines = f.readlines()
        except UnicodeDecodeError:
            with open(self._file_name, 'r', encoding=self.decoding_def) as f:
                self.__lines = f.readlines()

        self._file_size = len(self.__lines)

    def check_end_of_file(self):
        return self._pointer < len(self.__lines) - 1

    def find_elements_key(self):
        tmp = []
        for line in range(self._pointer, len(self.__lines)):
            if re.match(r'\W*\d.\d*E.\d\d', self.__lines[line]) is not None:
                time = self.__lines[line].strip()
                line += 2
                while re.match(r'\W*\d.\d*E.\d\d', self.__lines[line]) is None:
                    tmp.append(self.__lines[line].strip())
                    line += 1
                    if line == self._file_size: break
                self._pointer = line

                table = np.array([i.split() for i in tmp], float)
                sigma0 = table[:, 10]
                return sigma0, time


class ZasFile:
    def __init__(self):
        self._file_name = os.path.join(PROJECT_PATH, r'Zas.txt')
        self.decoding_def = locale.getpreferredencoding()

        try:
            with open(self._file_name, 'r', encoding='utf-8') as f:
                self.__lines = f.readlines()
        except UnicodeDecodeError:
            with open(self._file_name, 'r', encoding=self.decoding_def) as f:
                self.__lines = f.readlines()

        self.mbkg = float(self.__lines[2].strip().split()[5])
        self.mbkt = float(self.__lines[2].strip().split()[4])


def calc_N_beta(nf, shape):
    global KG, KT

    out = np.zeros(shape)
    XS = nf[:, 6].astype(float, copy=True)
    TK = nf[:, 3].astype(float, copy=True)
    R = nf[:, 1].astype(float, copy=True)

    for i in range(XS.shape[0]):
        if i == 0:
            out[i] = XS[i] * 4 * np.pi * TK[i] * (R[i] - 0) * 1.45e23 * KT * 0.8 / (3 * KG)
        else:
            out[i] = XS[i] * 4 * np.pi * TK[i] * (R[i] - R[i - 1]) * 1.45e23 * KT * 0.8 / (3 * KG)

    return out


def write_full(time, elem_table, par_table):
    tmp = ''
    tmp += time + '\n'
    tmp += '\t'.join([f'{i:14.14s}' for i in ELEMENTS_PATTERN.split()]) + '\n'
    for i in range(elem_table.shape[0]):
        tmp += '\t'.join([f'{v:14.8E}' for v in elem_table[i]]) + '\n'
    tmp += '\t'.join([f'{i:14.14s}' for i in NF_AFTER_PATTERN.split()[:5] + ['SIGMA0']]) + '\n'
    for i in range(par_table.shape[0]):
        tmp += '\t'.join([f'{v:14.8E}' for v in par_table[i]]) + '\n'
    tmp += '\n'
    return tmp


def write_strict(time, elem_table, par_table):
    tmp = ''

    header = r'r o n e o+ n+ o+2 n+2 o+3 n+3 o+4 n+4 O+5 N+5 O+6 N+6 O+7 N+7 O+8'.split()
    indexes = [ELEM_INDEX[key] for key in header]

    elem_table = elem_table.astype(float)

    new_elem_table = np.zeros((elem_table.shape[0], 19), dtype=float)
    for i in range(new_elem_table.shape[1]):
        new_elem_table[:, i] = elem_table[:, indexes[i]]

    # O = o + 2 * o2 + 3 * o3 + 1/2 * (o2+)
    new_elem_table[:, 1] = new_elem_table[:, 1] + \
                           2 * elem_table[:, ELEM_INDEX['o2']] + \
                           3 * elem_table[:, ELEM_INDEX['o3']] + \
                           0.5 * elem_table[:, ELEM_INDEX['o2+']]

    # N = n + 2 * n2 + 1/2 * (n2+)
    new_elem_table[:, 2] = new_elem_table[:, 2] \
                           + 2 * elem_table[:, ELEM_INDEX['n2']] \
                           + 0.5 * elem_table[:, ELEM_INDEX['n2+']]

    # O+ = 1/2 * (o2+) + (o+)
    new_elem_table[:, 4] = new_elem_table[:, 4] + 0.5 * elem_table[:, ELEM_INDEX['o+']]

    # N+ = 1/2 * (n2+) + (n+)
    new_elem_table[:, 5] = new_elem_table[:, 5] + 0.5 * elem_table[:, ELEM_INDEX['n2+']]

    # запись
    tmp += time + '\n'
    tmp += '\t'.join([f'{i:14.14s}' for i in header]) + '\n'
    for i in range(new_elem_table.shape[0]):
        tmp += '\t'.join([f'{v:14.8E}' for v in new_elem_table[i]]) + '\n'
    tmp += '\t'.join([f'{i:14.14s}' for i in NF_AFTER_PATTERN.split()[:5] + ['N_beta', 'SIGMA0']]) + '\n'
    for i in range(par_table.shape[0]):
        tmp += '\t'.join([f'{v:14.8E}' for v in par_table[i]]) + '\n'
    tmp += '\n'

    out_dict.update({time: {'elements': new_elem_table.tolist(), 'parameters': par_table.tolist()}})

    return tmp


def main(krm_path: str):
    global KG, KT, out_dict, PROJECT_PATH

    PROJECT_PATH = krm_path

    bf = BigFile()
    sf = SmallFile()
    zas = ZasFile()

    KG, KT = zas.mbkg, zas.mbkt

    out_dict = {'header': {
        'elements': ELEMENTS_PATTERN_FIXED.split(),
        'parameters': NF_AFTER_PATTERN.split()[:5] + ['N_beta', 'SIGMA0']
    }}
    out_string_full = ''
    out_string_fixed = ''

    while sf.check_end_of_file():
        sigma0, time = sf.find_elements_key()
        elem_table = bf.find_elements_key()
        _ = bf.find_elements_key()
        nf, time_bf = bf.find_nf_key()

        par_table = np.column_stack((nf, sigma0))

        out_string_full += write_full(time, elem_table, par_table)
        out_string_fixed += write_strict(time, elem_table, par_table)

    with open(os.path.join(PROJECT_PATH, 'results_full.txt'), 'w', encoding='utf-8') as wf:
        wf.write(out_string_full)
    with open(os.path.join(PROJECT_PATH, 'results.txt'), 'w', encoding='utf-8') as wf:
        wf.write(out_string_fixed)
    with open(os.path.join(PROJECT_PATH, 'results.json'), 'w') as file:
        json.dump(out_dict, file, indent=4, ensure_ascii=True)


if __name__ == '__main__':
    # for i, j, k in zip(ELEMENTS_PATTERN.split(), ELEMENTS_PATTERN_1.split(), ELEMENTS_PATTERN_2.split()):
    #     print(f'{i}  {j}  {k}  {i == j}')
    try:
        path = os.path.normpath(projectfilename)
    except Exception:
        path = input("Введите путь к папке проекта: ")

    if not os.path.exists(path):
        print(f'Не найдена папка {path}')
        exit(1)

    main(os.path.basename(path))
