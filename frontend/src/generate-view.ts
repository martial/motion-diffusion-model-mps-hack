// generate-view.ts
import { LitElement, html, css, unsafeCSS } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import styles from './styles/app.css?inline';

interface MotionData {
  sample_id: number;
  repetition_id: number;
  motion_data: string;
  parameters: string;
  text_prompt: string;
  motion_length: string;
  visualization: string;
}

@customElement('generate-view')
export class GenerateView extends LitElement {
  static styles = css`
    ${unsafeCSS(styles)}
  `;

  // State declarations
  @state()
  private isLoading = false;

  @state()
  private results: Array<Record<string, any>> = [];

  @state()
  private error: string | null = null;

  // Form handling methods
  private async handleSubmit(e: Event) {
    e.preventDefault();
    this.isLoading = true;
    this.error = null;
    
    try {
      const formData = new FormData(e.target as HTMLFormElement);
      const payload = {
        prompt: formData.get('prompt'),
        seed: parseInt(formData.get('seed') as string),
        num_samples: parseInt(formData.get('num_samples') as string),
        num_repetitions: 1,
        motion_length: parseFloat(formData.get('motion_length') as string),
        guidance_param: parseFloat(formData.get('guidance_param') as string),
      };

      const response = await fetch('http://localhost:3000/api/motion/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (data.status === 'success') {
        this.results = [data, ...this.results];
      } else {
        this.error = data.message;
      }
    } catch (e) {
      this.error = 'Failed to generate motion';
    } finally {
      this.isLoading = false;
    }
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const form = (e.target as HTMLElement).closest('form');
      if (form) {
        const submitEvent = new Event('submit', {
          bubbles: true,
          cancelable: true,
        });
        form.dispatchEvent(submitEvent);
      }
    }
  }

  private getFileUrl(file: string): string {
    const cleanPath = file.replace(/^.*outputs\//, '');
    return `http://localhost:3000/api/motion/outputs/${cleanPath}`;
  }

  private randomizeSeed() {
    const seedInput = document.querySelector('input[name="seed"]') as HTMLInputElement;
    if (seedInput) {
      seedInput.value = Math.floor(Math.random() * 1000000).toString();
    }
  }

  // Result rendering methods
  private renderResults() {
    return html`
      ${this.isLoading ? this.renderLoadingBar() : ''}
      ${this.results.length > 0 ? html`
        <div class="results-header">
          <button @click=${this.clearResults} class="clear-button">
            Clear Results
          </button>
        </div>
        ${this.results.map(result => this.renderGenerationResults(result))}
      ` : this.renderPlaceholder()}
    `;
  }

  private renderLoadingBar() {
    return html`
      <div class="loading-bar-container">
        <div class="loading-bar"></div>
      </div>
    `;
  }

  private renderPlaceholder() {
    return html`
      <div class="placeholder-container">
        <svg class="placeholder-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
            d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
            d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <div class="placeholder-text">Your generated motion will appear here</div>
      </div>
    `;
  }

  private renderGenerationResults(result: Record<string, any>) {
    return html`
      <div class="result-section" style="width: 100%;">
        <div class="info-section">
          <h3 class="label">Generation Info</h3>
          <div class="info-grid">
            <div class="info-item">
              <span class="info-label">Prompt</span>
              <span class="info-value">${result.parameters.prompt}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Samples</span>
              <span class="info-value">${result.parameters.num_samples}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Repetitions</span>
              <span class="info-value">${result.parameters.num_repetitions}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Duration</span>
              <span class="info-value">${result.parameters.motion_length}s</span>
            </div>
            <div class="info-item">
              <span class="info-label">Guidance</span>
              <span class="info-value">${result.parameters.guidance_param}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Seed</span>
              <span class="info-value">${result.parameters.seed}</span>
            </div>
          </div>
        </div>

        <div class="generations-grid" style="width: 100%;">
          ${result.files.data.map((data: MotionData) => html`
            <div class="generation-container" style="width: 340px;">
              <div class="video-container">
                <div class="video-header">
                  <span>Sample ${data.sample_id + 1} / Rep ${data.repetition_id + 1}</span>
                </div>
                <video 
                  controls 
                  loop 
                  autoplay 
                  muted 
                  class="video-preview"
                  src=${this.getFileUrl(data.visualization)}
                >
                </video>
              </div>
              
              <div class="download-group">
                <div class="download-group-files">
                  <a href=${this.getFileUrl(data.motion_data)} download class="download-button npy">
                    <svg class="download-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                    </svg>
                    Motion Data
                  </a>
                  <a href=${this.getFileUrl(data.parameters)} download class="download-button json">
                    <svg class="download-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                    </svg>
                    Parameters
                  </a>
                  <a href=${this.getFileUrl(data.visualization)} download class="download-button mp4">
                    <svg class="download-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                    </svg>
                    Video
                  </a>
                </div>
              </div>
            </div>
          `)}
        </div>
      </div>
    `;
  }

  // Main render method
  render() {
    return html`
      <div class="content-wrapper">
        <!-- Left Column: Form -->
        <div class="left-column">
          ${this.renderForm()}
        </div>
        
        <!-- Right Column: Results -->
        <div class="right-column">
          ${this.renderResults()}
        </div>
      </div>
    `;
  }

  // Template for the form section
  private renderForm() {
    return html`
      <div class="form-container">
        <form @submit=${this.handleSubmit}>
          <div class="form-group">
            <label class="label">Describe your motion</label>
            <textarea 
              name="prompt"
              required
              class="textarea"
              rows="4"
              @keydown=${this.handleKeyDown}
            >A person</textarea>
          </div>

          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div class="form-group">
              <label class="label">Seed</label>
              <div class="seed-input-group">
                <input type="number" name="seed" class="select" value="42" />
                <button 
                  type="button" 
                  class="randomize-button" 
                  @click=${this.randomizeSeed}
                >
                  Randomize
                </button>
              </div>
            </div>

            <div class="form-group">
              <label class="label">Number of Samples</label>
              <input type="number" name="num_samples" class="select" min="1" max="10" value="3" />
            </div>

            <div class="form-group">
              <label class="label">Motion Length (seconds)</label>
<input type="number" name="motion_length" class="select" min="1" max="15.68" step="0.5" value="6.0" />
            </div>

            <div class="form-group">
              <label class="label">Guidance Parameter</label>
              <input type="number" name="guidance_param" class="select" min="1" max="10" step="0.1" value="2.5" />
            </div>
          </div>

          <div class="form-group">
            <button type="submit" ?disabled=${this.isLoading} class="button">
              ${this.isLoading ? 'Generating...' : 'Generate Motion'}
            </button>
          </div>
        </form>

        ${this.error ? html`
          <div class="error-message">
            ${this.error}
          </div>
        ` : ''}
      </div>
    `;
  }

  private clearResults() {
    this.results = [];
  }
}