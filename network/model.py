import os
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from network import arch, callback, loss, optimizer, utils


class MyModel:
  def __init__(
    self,
    iterator,
    dataset,
    name,
    mode,
    loss_fn,
    optimizer_fn,
    batch_size,
    n_filters,
    augment,
    dataset_size,
    modality,
    class_weights,
    shapes,
    test_slices
  ):
    self.dataset = dataset
    self.checkpoint = '{}{}_{}_{}_b{}_f{}_a{}_{}_{}_{}'.format(
      name,
      mode,
      loss_fn,
      optimizer_fn,
      batch_size,
      n_filters,
      augment,
      dataset_size,
      modality,
      iterator
    )
    self.loss = loss_fn
    self.loss_fn = loss.get(loss_fn, utils.alpha)
    self.optimizer_fn = optimizer.get(optimizer_fn)
    self.model = arch.get(name + mode)(shapes[mode], n_filters)
    self.test_slices = test_slices


  def load(self):
    self.model.load_weights(f'output/{self.dataset}/models/{self.checkpoint}.hdf5')


  def compile(self):
    self.model.compile(
      loss=self.loss_fn,
      optimizer=self.optimizer_fn(),
      metrics=['accuracy', loss.dice, utils.alpha_coef(utils.alpha)]
    )


  def train(self, train_generator, valid_generator):
    checkpoint_file = f'output/{self.dataset}/models/{self.checkpoint}.hdf5'

    callbacks = [
      # TensorBoard(log_dir=f'output/{self.dataset}/logs/{self.checkpoint}'),
      ModelCheckpoint(checkpoint_file, monitor='val_dice', mode='max', save_best_only=True, verbose=1),
      EarlyStopping(min_delta=1e-5, monitor='val_dice', mode='max', patience=40, verbose=1),
      ReduceLROnPlateau(factor=0.5, min_lr=1e-6, monitor='val_dice', mode='max', patience=6, verbose=1)
    ]
    if self.loss in ['sdl', 'sgl']:
      callbacks.append(callback.AlphaScheduler(utils.alpha, utils.update_alpha))

    history = self.model.fit_generator(
      train_generator,
      epochs=100,
      callbacks=callbacks,
      validation_data=valid_generator,
      steps=
    )

    json_file = f'output/{self.dataset}/histories/{self.checkpoint}.json'
    output = pd.DataFrame(history.history)
    output.to_json(json_file)

    csv_file = f'output/{self.dataset}/histories/{self.checkpoint}.csv'
    output = pd.DataFrame(history.history['val_dice'])
    output.to_csv(csv_file)


  def evaluate_generator(self, generator):
    return self.model.evaluate_generator(generator, verbose=1)


  def predict_generator(self, generator):
    y_pred = self.model.predict_generator(generator, verbose=1)

    return (y_pred > 0.5).astype(np.uint8)

  
  def predict(self, x):
    y_pred = self.model.predict(x, verbose=1)

    return (y_pred > 0.5).astype(np.uint8)


  def summary(self):
    self.model.summary()


  def save_result(self, scenario, acc, dice, prec, rec, f1):
    csv_file = f'output/{self.dataset}/results.csv'
    result = [{
      'checkpoint': self.checkpoint,
      **scenario,
      'acc': round(acc, 4),
      'dice': round(dice, 4),
      'prec': round(prec, 4),
      'rec': round(rec, 4),
      'f1': round(f1, 4)
    }]
    output = pd.DataFrame(result)

    if not os.path.exists(csv_file):
      output.to_csv(csv_file, index=False, header=True, mode='a')
    else:
      output.to_csv(csv_file, index=False, header=False, mode='a')


  def save_visualization(self, x, y, y_pred):
    visualization_file = f'output/{self.dataset}/visualizations/{self.checkpoint}.png'
    y_comb = 2 * y + y_pred

    figure = utils.prepare_visualisation(x, y, y_pred, y_comb, self.test_slices)
    figure.savefig(visualization_file, bbox_inches='tight')


  def save_scan(self, x, y, y_pred):
    scan_file = f'output/{self.dataset}/scans/{self.checkpoint}.npy'
    y_comb = 2 * y + y_pred

    np.save(scan_file, y_comb)
