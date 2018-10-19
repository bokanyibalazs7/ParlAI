from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
import os, copy




class CNNDMTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'cnn_dm'

        opt['datafile'] = self._path(opt)
        self.opt = copy.deepcopy(opt)

        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)

        return os.path.join(opt['datapath'], 'CNN_DM')

    def setup_data (self, input_path):

        print('loading: ' + input_path)
        paths = [os.path.join(input_path, 'cnn', 'stories'), os.path.join(input_path, 'dailymail', 'stories')]

        def extract_data_and_labels(text):
            text_sections = text.split('@highlight')
            return text_sections[0], [text.strip('\n') for text in text_sections[1:]]

        self.question = 'What is the summary?'

        new_episode = True
        for path in paths:
            for file in os.listdir(path):
                if file.endswith('.story'):
                    with open(os.path.join(path,file)) as file_data:
                        data, label = extract_data_and_labels(file_data.read())

                    yield (data + '\n' + self.question, label, None, None), new_episode

class DefaultTeacher(CNNDMTeacher):
    pass

