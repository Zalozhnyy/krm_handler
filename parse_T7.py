import os

from parse_results import BigFile as Parser
import parse_results


def get_t7_data(krm_path: str):
    parse_results.PROJECT_PATH = krm_path

    out_dict = {'header': parse_results.NF_AFTER_PATTERN.split(),
                'data': {}}

    parser = Parser()

    while parser.check_end_of_file():
        try:
            _ = parser.find_elements_key()
            _ = parser.find_elements_key()
            nf, time_bf = parser.find_nf_key()

            out_dict['data'].update({time_bf: nf})

        except EOFError:
            break

    return out_dict

