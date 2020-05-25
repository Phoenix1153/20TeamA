import config
from RC.preproc import Preproc
from collections import Counter
import ujson as json

class RCTest():
    def __init__(self):
        super().__init__()
        self.pproc = Preproc()

    def test_process_data(self,config):
        word_counter = Counter()
        char_counter = Counter()
        s={"data":[{"title":"Super_Bowl_50","paragraphs":[{"context":"this is a test context","qas":[{"answers":[{"answer_start":0,"text":"this is"}],"question":"is this a test context?","id":"randomid0001"}]}]}],"version":"test_simple"}
        with open("test_tmp_rawdata_file.json",'w') as fh:
            json.dump(s,fh)

        train_examples, train_eval = self.pproc.process_file("test_tmp_rawdata_file.json", "unitText", word_counter, char_counter)

        emb_mat, word2idx_dict =  self.pproc.get_embedding(word_counter, "unitText",vec_size=config.glove_dim)
        emb_mat, char2idx_dict =  self.pproc.get_embedding(char_counter, "unitText",vec_size=config.char_dim)

        self.pproc.build_features(config, train_examples, "train", "test_tmp_record_file.json",word2idx_dict, char2idx_dict)
        
        print(train_eval)
        print(train_examples)
        print(word2idx_dict)
        print(char2idx_dict)

if __name__ == '__main__':
    r=RCTest()
    r.test_process_data(config)