// app-element.ts
import { LitElement, html, css, unsafeCSS } from 'lit'
import { customElement } from 'lit/decorators.js'
import { Router } from '@lit-labs/router'
import styles from './styles/app.css?inline';
import './generate-view'
import './nav-element'
import './list-view'

@customElement('app-element')
export class AppElement extends LitElement {
  static styles = css`
    ${unsafeCSS(styles)}
  `;

  private router = new Router(this, [
    {
      path: '/',
      render: () => html`<generate-view></generate-view>`
    },
    {
      path: '/generate',
      render: () => html`<generate-view></generate-view>`
    },
    {
      path: '/list',
      render: () => html`<list-view></list-view>`
    }
  ]);

  render() {
    return html`
      <div class="app-container">
        <nav-bar></nav-bar>
        <div class="main-content">
          <header>
            <h1 class="header-title">LIMINAL MDM</h1>
          </header>
          <main>
            ${this.router.outlet()}
          </main>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'app-element': AppElement
  }
}