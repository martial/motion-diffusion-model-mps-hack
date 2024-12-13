// nav-bar.ts
import { LitElement, html, css, unsafeCSS } from 'lit';
import { customElement } from 'lit/decorators.js';
import styles from './styles/app.css?inline';

@customElement('nav-bar')
export class NavBar extends LitElement {
  static styles = css`
    ${unsafeCSS(styles)}
  `;

  render() {
    return html`
      <div class="side-nav">
        <nav class="nav-links">
          <a href="/generate" class="nav-link ${location.pathname === '/generate' || location.pathname === '/' ? 'active' : ''}">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
            </svg>
          </a>
          <a href="/list" class="nav-link ${location.pathname === '/list' ? 'active' : ''}">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
            </svg>
          </a>
        
        </nav>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'nav-bar': NavBar
  }
}