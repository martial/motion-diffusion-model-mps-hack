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
    
    .left-column {
      max-height: 100vh;
      overflow-y: auto;
      padding-right: 16px;
    }

    .prompt-inputs {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .direct-input, .ai-input {
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .ai-prompts-container {
      margin: 1rem 0;
      padding: 2rem;
      background: rgb(30, 30, 30);
      border-radius: 16px;
      border: 1px solid rgb(45, 45, 45);
    }

    .ai-prompts-container h3 {
      color: white;
      font-size: 1.5rem;
      margin: 0 0 1rem 0;
    }

    .ai-prompts-list {
      list-style: none;
      padding: 0;
      margin: 1rem 0;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .ai-prompts-list li {
      padding: 1rem;
      background: transparent;
      border-radius: 8px;
      border: 1px solid rgb(63, 93, 255);
      color: rgb(63, 93, 255);
      font-size: 1rem;
      line-height: 1.5;
    }

    .progress-bar {
      width: 100%;
      height: 4px;
      background: rgb(45, 45, 45);
      border-radius: 2px;
      margin-top: 1rem;
    }

    .progress {
      height: 100%;
      background: rgb(63, 93, 255);
      border-radius: 2px;
      transition: width 0.3s ease;
    }

    .button.secondary {
      background: #6c757d;
    }

    .button.secondary:hover {
      background: #5a6268;
    }

    .error-message {
      margin-top: 1rem;
      padding: 1rem;
      background-color: #ffebee;
      color: #c62828;
      border-radius: 4px;
      border: 1px solid #ef9a9a;
    }

    button.primary {
      width: 100%;
      padding: 1rem;
      border-radius: 8px;
      background: rgb(63, 93, 255);
      color: white;
      border: none;
      font-size: 1rem;
      cursor: pointer;
      margin-top: 1rem;
    }

    button.primary:hover {
      background: rgb(71, 99, 255);
    }

    .placeholder-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      text-align: center;
      color: #666;
    }

    .placeholder-icon {
      width: 48px;
      height: 48px;
      margin-bottom: 1rem;
    }

    .placeholder-text {
      font-size: 1.1rem;
    }

    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }

    .loading .placeholder-icon {
      animation: spin 2s linear infinite;
    }

    .loading .placeholder-text {
      color: transparent;
      background: linear-gradient(
        90deg, 
        rgba(63, 93, 255, 0.3) 25%, 
        rgba(63, 93, 255, 0.8) 50%, 
        rgba(63, 93, 255, 0.3) 75%
      );
      background-size: 200% auto;
      background-clip: text;
      -webkit-background-clip: text;
      animation: shimmer 2s linear infinite;
    }

    @keyframes indeterminate {
      0% {
        transform: translateX(-100%);
      }
      100% {
        transform: translateX(200%);
      }
    }

    .progress.indeterminate {
      width: 50% !important;
      animation: indeterminate 1.5s ease-in-out infinite;
    }

    @keyframes shimmer {
      0% {
        background-position: 200% center;
      }
      100% {
        background-position: -200% center;
      }
    }
  `;

  // State declarations
  @state()
  private isLoading = false;

  @state()
  private results: Array<Record<string, any>> = [];

  @state()
  private error: string | null = null;

  @state()
  private aiPrompts: string[] = [];

  @state()
  private isGeneratingBatch = false;

  @state()
  private batchProgress = 0;

  @state()
  private lastFocusedInput: 'direct' | 'ai' = 'direct';

  @state()
  private activeAction: 'generate' | 'ai-prompts' | 'batch' | 'randomize' | null = null;

  // Form handling methods
  private async handleSubmit(e: Event) {
    e.preventDefault();
    this.activeAction = 'generate';
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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        this.results = [data, ...this.results];
      } else {
        this.error = data.message || 'Failed to generate motion';
      }
    } catch (e) {
      this.error = e instanceof Error ? e.message : 'Failed to generate motion';
    } finally {
      this.isLoading = false;
      this.activeAction = null;
    }
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      
      if (this.lastFocusedInput === 'direct') {
        const form = (e.target as HTMLElement).closest('form');
        if (form) {
          const submitEvent = new Event('submit', {
            bubbles: true,
            cancelable: true,
          });
          form.dispatchEvent(submitEvent);
        }
      } else if (this.lastFocusedInput === 'ai') {
        this.generateAIPrompts(e);
      }
    }
  }

  private getFileUrl(file: string): string {
    const cleanPath = file.replace(/^.*outputs\//, '');
    return `http://localhost:3000/api/motion/outputs/${cleanPath}`;
  }

  private async generateAIPrompts(e: Event) {
    e.preventDefault();
    this.activeAction = 'ai-prompts';
    this.isLoading = true;
    this.error = null;
    
    try {
      // Find the closest form element from the button
      const form = this.renderRoot.querySelector('form');
      if (!form) {
        throw new Error('Form not found');
      }
      
      const formData = new FormData(form);
      const inputText = formData.get('ai_input');
      
      if (!inputText) {
        throw new Error('Please enter text for AI prompts generation');
      }

      const response = await fetch('http://localhost:3000/api/motion/ai-batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText })
      });

      const data = await response.json();
      if (data.status === 'success') {
        this.aiPrompts = data.prompts;
      } else {
        this.error = data.message;
      }
    } catch (e) {
      this.error = e instanceof Error ? e.message : 'Failed to generate AI prompts';
    } finally {
      this.isLoading = false;
      this.activeAction = null;
    }
  }

  private async randomizeSeed() {
    this.activeAction = 'randomize';
    this.isLoading = true;
    const seedInput = this.renderRoot.querySelector('input[name="seed"]') as HTMLInputElement;
    if (seedInput) {
      seedInput.value = Math.floor(Math.random() * 1000000).toString();
    }
    await new Promise(resolve => setTimeout(resolve, 200));
    this.isLoading = false;
    this.activeAction = null;
  }

  private async generateBatchMotions() {
    this.activeAction = 'batch';
    this.isLoading = true;
    this.batchProgress = 0;
    
    for (let i = 0; i < this.aiPrompts.length; i++) {
      const prompt = this.aiPrompts[i];
      try {
        const formData = new FormData(this.renderRoot.querySelector('form') as HTMLFormElement);
        const payload = {
          prompt: prompt,
          seed: parseInt(formData.get('seed') as string),
          num_samples: parseInt(formData.get('num_samples') as string),
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
        }
      } catch (e) {
        console.error(`Failed to generate motion for prompt: ${prompt}`);
      }
      this.batchProgress = ((i + 1) / this.aiPrompts.length) * 100;
    }
    
    this.isLoading = false;
    this.activeAction = null;
  }

  // Result rendering methods
  private renderResults() {
    return html`
      ${(this.activeAction === 'generate' || this.activeAction === 'batch') ? html`
        <div class="progress-bar" style="margin: 0 0 1rem 0;">
          <div class="progress ${this.activeAction === 'generate' ? 'indeterminate' : ''}" 
               style="width: ${this.activeAction === 'batch' ? this.batchProgress : 100}%">
          </div>
        </div>
        ${this.renderLoadingPlaceholder()}
      ` : ''}
      ${this.results.length > 0 ? html`
        <div class="results-header">
          <button @click=${this.clearResults} class="clear-button">
            Clear Results
          </button>
        </div>
        ${this.results.map(result => this.renderGenerationResults(result))}
      ` : !this.isLoading ? this.renderPlaceholder() : ''}
    `;
  }

  private renderLoadingPlaceholder() {
    const currentSample = Math.floor(this.batchProgress / (100 / this.results.length)) + 1;
    const totalSamples = this.activeAction === 'batch' ? this.aiPrompts.length : 1;
    
    return html`
      <div class="placeholder-container loading" style="padding: 1rem; margin-bottom: 1rem;">
      
        <div class="placeholder-text">
          Making the jazz happen ${currentSample}/${totalSamples}
        </div>
      </div>
    `;
  }

  private renderPlaceholder() {
    return html`
      <div class="placeholder-container ${this.isLoading ? 'loading' : ''}">
          ${this.isLoading ? html`
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
              d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" 
              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          ` : html`
          
          `}
        </svg>
        <div class="placeholder-text">
          ${this.isLoading ? 'Generating your motion...' : 'Your generated motion will appear here'}
        </div>
      </div>
    `;
  }

  private renderGenerationResults(result: Record<string, any>) {
    return html`
      <div class="result-section" style="width: 100%;">
        <div class="info-section">
          <h3 class="label">${result.parameters.prompt}</h3>
          <div class="info-grid">
          
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
                  <a href=${this.getFileUrl(data.motion_data)} download target="_blank" class="download-button npy">
                    <svg class="download-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                    </svg>
                    Motion Data
                  </a>
                  <a href=${this.getFileUrl(data.parameters)} download target="_blank" class="download-button json">
                    <svg class="download-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                    </svg>
                    Parameters
                  </a>
                  <a href=${this.getFileUrl(data.visualization)} download target="_blank" class="download-button mp4">
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
            <div class="prompt-inputs">
              <div class="direct-input">
                <textarea 
                  name="prompt"
                  required
                  class="textarea"
                  rows="4"
                  @keydown=${this.handleKeyDown}
                  @focus=${this.handleFocus}
                >A person</textarea>
                <button type="submit" ?disabled=${this.isLoading} class="button primary">
                  ${this.activeAction === 'generate' ? 'Generating...' : 'Generate Motion'}
                </button>
              </div>
              
              <div class="ai-input">
                <textarea 
                  name="ai_input"
                  class="textarea"
                  rows="4"
                  placeholder="Or describe multiple motions for AI to generate variants..."
                  @keydown=${this.handleKeyDown}
                  @focus=${this.handleFocus}
                ></textarea>
                <button 
                  type="button" 
                  class="button primary" 
                  @click=${this.generateAIPrompts}
                  ?disabled=${this.isLoading}
                >
                  ${this.activeAction === 'ai-prompts' ? 'Generating...' : 'Generate AI Prompts'}
                </button>
              </div>
            </div>
          </div>

          ${this.aiPrompts.length > 0 ? html`
            <div class="ai-prompts-container">
              <h3>Generated Prompts:</h3>
              <ul class="ai-prompts-list">
                ${this.aiPrompts.map(prompt => html`
                  <li>${prompt}</li>
                `)}
              </ul>
              <button 
                type="button"
                @click=${this.generateBatchMotions} 
                ?disabled=${this.isLoading}
                class="button primary"
              >
                ${this.activeAction === 'batch' ? 'Generating...' : 'Generate All Motions'}
              </button>
              ${this.isLoading ? html`
                <div class="progress-bar">
                  <div class="progress" style="width: ${this.batchProgress}%"></div>
                </div>
              ` : ''}
            </div>
          ` : ''}

          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div class="form-group">
              <label class="label">Seed</label>
              <div class="seed-input-group">
                <input type="number" name="seed" class="select" value="42" />
                <button 
                  type="button" 
                  class="randomize-button" 
                  @click=${this.randomizeSeed}
                  ?disabled=${this.isLoading}
                >
                  ${this.activeAction === 'randomize' ? 'Randomizing...' : 'Randomize'}
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

  private handleFocus(e: FocusEvent) {
    const textarea = e.target as HTMLTextAreaElement;
    this.lastFocusedInput = textarea.name === 'prompt' ? 'direct' : 'ai';
  }
}