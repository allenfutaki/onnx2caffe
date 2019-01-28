import os
import sys
import shutil

from pyhocon import ConfigFactory


class Option(object): 
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)

        # ------------- general options ----------------------------------------
        self.experiment_id = self.conf['experiment_id']
        self.save_path = self.conf['save_path']  # log path
        self.onnxmodel = self.conf['onnx']
        self.caffemodel = self.conf['caffemodel']  # path for loading data set
        self.prototxt = self.conf['prototxt']

        self.set_save_path()

    def set_save_path(self):

        self.save_path = os.path.join(self.save_path,
                             "convert_{}/".format(
                self.experiment_id))
        

        if os.path.exists(self.save_path):
            print("{} file exists!".format(self.save_path))
            action = input("d (delete) / q (quit):").lower().strip()
            act = action
            if act == 'd':
                shutil.rmtree(self.save_path)
            elif act == 'q':
                sys.exit(1)
            else:
                raise ValueError("Unknown command.")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
