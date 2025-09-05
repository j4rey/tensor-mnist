import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'tensor-mnist';

  links = [
    {
      title: 'Home',
      url: '/'
    },
    {
      title: 'Train',
      url: '/train'
    },
    {
      title: 'Predict',
      url: '/predict'
    }
  ]
}
