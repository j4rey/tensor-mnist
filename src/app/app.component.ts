import { Component } from '@angular/core';

import { RouterOutlet } from '@angular/router';

@Component({
    selector: 'app-root',
    imports: [RouterOutlet],
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
