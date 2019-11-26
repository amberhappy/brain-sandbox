import argparse

from network import extractor, generator, model, utils
import config


parser = argparse.ArgumentParser(
  prog='brain-sandbox',
  description='Train neural network or load model and make predictions'
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-t', '--train', action='store_true', help='train neural network')
group.add_argument('-e', '--evaluate', action='store_true', help='load model and make predictions')
parser.add_argument('-x', '--extract', action='store_true', help='extract a dataset')
parser.add_argument('-l', '--log', action='store_true', help='log results')
parser.add_argument('-i', '--iterator', type=int, default=0, help='select iterator number')
parser.add_argument('-s', '--scenario',
  type=int,
  choices=range(-len(config.scenarios), len(config.scenarios)),
  default=-1,
  help='select a scenario number'
)
parser.add_argument('-d', '--dataset',
  type=int,
  choices=range(len(config.datasets)),
  default=1,
  help='select a dataset number'
)
args = parser.parse_args()


dataset = config.datasets[args.dataset]

if args.extract:
  my_extractor = extractor.MyExtractor(
    name=dataset['name'],
    path=dataset['path'],
    scan_paths=dataset['scan_paths'],
    mask_path=dataset['mask_path'],
    labels=dataset['labels'],
    modes=dataset['modes'],
    extensions=dataset['extensions'],
    shapes=dataset['shapes'],
    extraction_sizes=dataset['extraction_sizes'],
    dirs=dataset['dirs']
  )

  my_extractor.extract_dataset()


scenario = config.scenarios[args.scenario]

x_train, y_train, x_valid, y_valid, x_test, y_test = utils.create_paths(
  dataset['name'], scenario['mode'], scenario['dataset'], scenario['modality']
)

class_weights = utils.calculate_weights(
  dataset['name'], scenario['mode'], scenario['dataset'], scenario['modality'], dataset['shapes']
)

train_generator = generator.DataSequence(
  x_train, y_train, dataset['shapes'], scenario['mode'], scenario['batch_size'], scenario['augment']
)
valid_generator = generator.DataSequence(
  x_valid, y_valid, dataset['shapes'], scenario['mode'], scenario['batch_size'], scenario['augment']
)
test_generator = generator.DataSequence(
  x_test, y_test, dataset['shapes'], scenario['mode'], scenario['batch_size'], shuffle=False
)

my_model = model.MyModel(
  iterator=args.iterator,
  dataset=dataset['name'],
  name=scenario['arch'],
  mode=scenario['mode'],
  loss_fn=scenario['loss_fn'],
  optimizer_fn=scenario['optimizer_fn'],
  batch_size=scenario['batch_size'],
  n_filters=scenario['n_filters'],
  augment=scenario['augment'],
  dataset_size=scenario['dataset'],
  modality=scenario['modality'],
  class_weights=class_weights,
  shapes=dataset['shapes'],
  test_slices=dataset['test_slices']
)

my_model.compile()

if args.train:
  my_model.train(train_generator, valid_generator)

my_model.load()

acc = my_model.evaluate_generator(test_generator)[1]
dice = my_model.evaluate_generator(test_generator)[2]

x, y = utils.extract_generator(test_generator)

y_pred = my_model.predict_generator(test_generator)

x, y, y_pred = utils.squeeze_all(x, y, y_pred)

if scenario['mode'] is '3d':
  x, y, y_pred = utils.uncubify_all(x, y, y_pred)

tp, fn, fp, tn = utils.calc_conf_matrix(y, y_pred)

prec, rec, f1 = utils.calc_metrics(y, y_pred)

print('prec: ', round(prec, 4), ', rec: ', round(rec, 4), ', f1: ', round(f1, 4))

if args.log:
  my_model.save_result(scenario, acc, dice, prec, rec, f1)
  my_model.save_visualization(x, y, y_pred)
