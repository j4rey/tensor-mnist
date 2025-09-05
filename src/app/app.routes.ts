import { Routes } from '@angular/router';

export const routes: Routes = [
    {
        path: '',
        loadComponent: () => import('./home/home.component').then(m => m.HomeComponent)
    },
    {
        path: 'train',
        loadComponent: () => import('./train/train.component').then(m => m.TrainComponent)
    },
    {
        path: 'predict',
        loadComponent: () => import('./predict/predict.component').then(m => m.PredictComponent)
    }
];
