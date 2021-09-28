import numpy as np
import pickle
import sys
from absl import app
from absl import flags

model_names=['basicModel_f_lbs_10_207_0_v1.0.0.pkl', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl']
name_mapper={'basicModel_f_lbs_10_207_0_v1.0.0.pkl':'SMPL_FEMALE.pkl', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl':'SMPL_MALE.pkl'}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'model_path',
    'smpl/models/',
    'input basice model which download from webpage')

def main(_):
    src_path = FLAGS.model_path
    print(src_path)
    if src_path[-1]!='/':
        src_path+='/'
    for model_name in model_names:
        with open(src_path+model_name, 'rb') as f:
            src_data = pickle.load(f, encoding="latin1")
            model = {
                'J_regressor': src_data['J_regressor'],
                'weights': np.array(src_data['weights']),
                'posedirs': np.array(src_data['posedirs']),
                'v_template': np.array(src_data['v_template']),
                'shapedirs': np.array(src_data['shapedirs']),
                'f': np.array(src_data['f']),
                'kintree_table': src_data['kintree_table']
            }
            if 'cocoplus_regressor' in src_data.keys():
                model['joint_regressor'] = src_data['cocoplus_regressor']
        with open(src_path + name_mapper[model_name], 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    app.run(main)