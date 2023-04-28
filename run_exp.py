from train_asam import main
import os
import glob
import shutil

path = '/home/hossein/projects/hifu/FCBFormer_V1/Data/HIFU_data/'

def remove_all():
    files = glob.glob('/home/hossein/projects/hifu/FCBFormer_V1/Data/HIFU_data//test_data_asam/images/before/*.jpg')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))



for (root, subdirs, files) in os.walk(os.path.join(path, 'test_data_tmp', 'images', 'before')):
    for file in files:
        if '.jpg' in file:
            remove_all()
            src = os.path.join(root, file)
            dst = os.path.join(path, 'test_data_asam',  'images', 'before', file)
            shutil.copyfile(src, dst)

            to_remove = os.path.join(path,  'images', 'before', file)
            if os.path.exists(to_remove):
                os.remove(to_remove)

            # call train
            run_name = file.split('.')[0]
            main(run_name)
            # bring the image back
            src = os.path.join(path, 'test_data_tmp', 'images', 'before', file)
            dst = os.path.join(path,  'images', 'before', file)
            shutil.copyfile(src, dst)

