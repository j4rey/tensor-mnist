import { Component, Inject } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { DataService } from '../services/data';

@Component({
  selector: 'app-predict',
  standalone: true,
  imports: [],
  templateUrl: './predict.component.html',
  styleUrl: './predict.component.scss'
})
export class PredictComponent {

  readonly classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

  constructor(@Inject(DataService) private data: DataService) {}

  async ngAfteVrViewInit() { 
    await this.data.load();

    const model = this.getModel();

    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);
    await this.train(model, this.data);
    await this.showAccuracy(model, this.data);
    await this.showConfusion(model, this.data);

    await model.save('download://mnist-model');
  }

  getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    model.add(
      tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 3,
        filters: 32,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );

    model.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        //strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );


    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] })); //, strides: [2, 2]

    model.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        //strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );

    model.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        //strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));// strides: [2, 2]

    model.add(tf.layers.flatten());

    model.add(
      tf.layers.dense({
        units: 256,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );

    model.add(
      tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling',
      })
    );

    return model;
  }

  async train(model: any, data: any) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels,
      ];
    });

    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels,
      ];
    });

    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 7,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

  async doPrediction(model: any, data: any, TEST_DATA_SIZE = 500) {
    const testData = data.nextTestBatch(TEST_DATA_SIZE);
    const testXs = testData.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]);
    const labels = testData.labels.argMax([-1]);

    const preds = model.predict(testXs).argMax([-1]);

    testXs.dispose();
    return [preds, labels];
  }

  async showAccuracy(model: any, data: any) {
    const [preds, labels] = await this.doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, this.classNames);

    labels.dispose();
    preds.dispose();
  }

  async showConfusion(model: any, data: any) {
    const [preds, labels] = await this.doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(
      container,
      { values: confusionMatrix , tickLabels: this.classNames },
    );

    labels.dispose();
    preds.dispose();
  }
}
