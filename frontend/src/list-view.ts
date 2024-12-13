import { LitElement, html, css, unsafeCSS } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import styles from './styles/app.css?inline';

interface GenerationParams {
  dataset: string;
  fps: number;
  guidance_param: number;
  motion_length: number;
  n_frames: number;
  prompt: string;
  seed: number;
}

interface MotionData {
  generation_params: GenerationParams;
  motion_data: string;
  motion_length: string;
  parameters: string;
  repetition_id: number;
  sample_id: number;
  text_prompt: string;
  visualization: string;
}

interface MotionFile {
  visualizations: string[];
  data: MotionData[];
}

interface Motion {
  id: string;
  timestamp: number;
  files: MotionFile;
}

@customElement('list-view')
export class ListView extends LitElement {
    static styles = css`
    ${unsafeCSS(styles)}
    
    .filter-container {
      margin: 2rem 0;
      padding: 0 24px;
      display: flex;
      gap: 16px;
      align-items: center;
    }
  
    select {
      padding: 12px 16px;
      font-size: 0.875rem;
      border: 1px solid #1E293B;
      border-radius: 8px;
      background-color: #0B1120;
      color: #94A3B8;
      cursor: pointer;
      appearance: none;
      background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394A3B8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
      background-repeat: no-repeat;
      background-position: right 12px center;
      background-size: 16px;
      min-width: 180px;
      transition: all 0.2s ease;
    }
  
    select:hover {
      background-color: #151F32;
      border-color: #2D3748;
      color: #F8FAFC;
    }
  
    select:focus {
      outline: none;
      border-color: #2563EB;
      color: #F8FAFC;
    }
  
    select option {
      background-color: #0B1120;
      color: #94A3B8;
      padding: 12px;
    }
  `;

  @state()
  private motions: Motion[] = [];

  @state()
  private isLoading = true;

  @state()
  private error: string | null = null;

  @state()
  private selectedDate: string = 'all';

  @state()
  private selectedPrompt: string = 'all';

  connectedCallback() {
    super.connectedCallback();
    this.loadMotions();
  }

  private async loadMotions() {
    try {
      const response = await fetch('http://localhost:3000/api/motion/list');
      const data = await response.json();
      
      if (data.status === 'success') {
        this.motions = data.motions;
      } else {
        this.error = data.message;
      }
    } catch (e) {
      this.error = 'Failed to load motions';
    } finally {
      this.isLoading = false;
    }
  }

  private formatTimestamp(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleString();
  }

  private getFileUrl(file: string): string {
    return `http://localhost:3000/api/motion/outputs/${file}`;
  }

  private groupMotionsByDay(motions: Motion[]): Map<string, Motion[]> {
    const groups = new Map<string, Motion[]>();
    
    motions
      .filter(motion => motion.files.data[0]?.generation_params?.prompt)
      .forEach(motion => {
        const date = new Date(motion.timestamp * 1000).toLocaleDateString();
        if (!groups.has(date)) {
          groups.set(date, []);
        }
        groups.get(date)?.push(motion);
      });
    
    return groups;
  }

  private getAvailableDates(): string[] {
    const dates = new Set<string>();
    this.motions.forEach(motion => {
      const date = new Date(motion.timestamp * 1000).toLocaleDateString();
      dates.add(date);
    });
    return Array.from(dates).sort().reverse(); // Most recent first
  }

  private getAvailablePrompts(): string[] {
    const prompts = new Set<string>();
    this.motions.forEach(motion => {
      const prompt = motion.files.data[0]?.generation_params?.prompt;
      if (prompt) prompts.add(prompt);
    });
    return Array.from(prompts).sort();
  }

  private filterMotions(motions: Motion[]): Motion[] {
    return motions
      .filter(motion => motion.files.data[0]?.generation_params?.prompt)
      .filter(motion => {
        if (this.selectedPrompt === 'all') return true;
        const prompt = motion.files.data[0]?.generation_params?.prompt;
        return prompt === this.selectedPrompt;
      });
  }

  render() {
    if (this.isLoading) {
      return html`<div class="loading-container">Loading...</div>`;
    }

    if (this.error) {
      return html`<div class="error-message">${this.error}</div>`;
    }

    const dates = this.getAvailableDates();
    const prompts = this.getAvailablePrompts();
    const filteredMotions = this.filterMotions(this.motions);
    const motionsByDay = this.groupMotionsByDay(filteredMotions);
    const filteredEntries = Array.from(motionsByDay.entries())
      .filter(([date]) => this.selectedDate === 'all' || date === this.selectedDate);

    return html`
      <div class="filter-container">
        <select @change=${(e: Event) => this.selectedDate = (e.target as HTMLSelectElement).value}>
          <option value="all">All Days</option>
          ${dates.map(date => html`
            <option value=${date} ?selected=${this.selectedDate === date}>${date}</option>
          `)}
        </select>

        <select @change=${(e: Event) => this.selectedPrompt = (e.target as HTMLSelectElement).value}>
          <option value="all">All Prompts</option>
          ${prompts.map(prompt => html`
            <option value=${prompt} ?selected=${this.selectedPrompt === prompt}>${prompt}</option>
          `)}
        </select>
      </div>

      <div class="generations-list">
        ${filteredEntries.map(([date, dayMotions]) => html`
          <div class="day-group">
            <h2 class="day-header">${date}</h2>
            ${dayMotions.map(motion => {
              const params = motion.files.data[0]?.generation_params;
              return html`
                <div class="generation-container">
                  <div class="generation-header">
                    <div class="generation-info">
                      <div class="generation-prompt">${params?.prompt || 'No prompt'}</div>
                      <div class="generation-params">
                        <span>Length: ${params?.motion_length}s</span>
                        <span>FPS: ${params?.fps}</span>
                        <span>Frames: ${params?.n_frames}</span>
                        <span>Guidance: ${params?.guidance_param}</span>
                        <span>Seed: ${params?.seed}</span>
                      </div>
                    </div>
                    <span class="motion-timestamp">${this.formatTimestamp(motion.timestamp)}</span>
                  </div>
                  <div class="samples-grid">
                    ${motion.files.data.map(data => html`
                      <div class="video-container">
                        <div class="video-header">
                          <span>Sample ${data.sample_id + 1} / Rep ${data.repetition_id + 1}</span>
                        </div>
                        <video 
                          controls 
                          loop 
                          muted 
                          class="video-preview"
                          src=${this.getFileUrl(data.visualization)}
                        ></video>
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
            })}
          </div>
        `)}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'list-view': ListView
  }
}
