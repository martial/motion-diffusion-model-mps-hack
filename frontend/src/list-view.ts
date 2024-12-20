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
  smpl_data?: string;
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
  
    .download-button.smpl {
      background-color: #8e44ad;
    }
  
    a.download-button.smpl {
      background-color: #2ecc71;
    }
  
    a.download-button.smpl:hover {
      background-color: #27ae60;
    }
  
    button.download-button.smpl:hover {
      background-color: #9b59b6;
    }
  
    .download-button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  
    @keyframes gradient {
      0% {
        background-position: 0% 50%;
      }
      50% {
        background-position: 100% 50%;
      }
      100% {
        background-position: 0% 50%;
      }
    }
  
    .download-button.smpl.generating {
      background: linear-gradient(-45deg, #8e44ad, #9b59b6, #3498db, #2980b9);
      background-size: 400% 400%;
      animation: gradient 3s ease infinite;
      position: relative;
      overflow: hidden;
    }
  
    .download-button.smpl.generating::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      height: 2px;
      background: rgba(255, 255, 255, 0.5);
      animation: loading 2s linear infinite;
    }
  
    @keyframes loading {
      0% {
        transform: translateX(-100%);
      }
      100% {
        transform: translateX(100%);
      }
    }
  
    .action-buttons {
      display: flex;
      gap: 8px;
      position: static;
      top: auto;
      right: auto;
    }
  
    .action-button {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid #1E293B;
      border-radius: 6px;
      padding: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
      color: #94A3B8;
    }
  
    .action-button:hover {
      background: rgba(30, 41, 59, 0.8);
      color: #F8FAFC;
    }
  
    .action-button.favorite {
      color: #94A3B8;
    }
  
    .action-button.favorite.active {
      color: #EAB308;
    }
  
    .action-button.remove:hover {
      background: rgba(220, 38, 38, 0.8);
      border-color: #DC2626;
    }
  
    .generation-container {
      position: relative;
    }
  
    .video-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px;
      background: rgba(15, 23, 42, 0.8);
    }
  
    .filter-button {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      font-size: 0.875rem;
      border: 1px solid #1E293B;
      border-radius: 8px;
      background-color: #0B1120;
      color: #94A3B8;
      cursor: pointer;
      transition: all 0.2s ease;
    }
  
    .filter-button:hover {
      background-color: #151F32;
      border-color: #2D3748;
      color: #F8FAFC;
    }
  
    .filter-button.favorite.active {
      background-color: #1E293B;
      border-color: #EAB308;
      color: #EAB308;
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

  @state()
  private generatingSMPL: Record<string, boolean> = {};

  @state()
  private favorites: Set<string> = new Set();

  @state()
  private showFavoritesOnly = false;

  @state()
  private hidden: Set<string> = new Set();

  async connectedCallback() {
    super.connectedCallback();
    await this.loadHidden();
    this.loadMotions();
    await this.loadFavorites();
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

  private async loadFavorites() {
    try {
        const response = await fetch('http://localhost:3000/api/motion/favorites');
        const data = await response.json();
        if (data.status === 'success') {
            this.favorites = new Set(data.favorites);
        }
    } catch (e) {
        this.error = 'Failed to load favorites';
    }
  }

  private async loadHidden() {
    try {
        const response = await fetch('http://localhost:3000/api/motion/hidden');
        const data = await response.json();
        if (data.status === 'success') {
            this.hidden = new Set(data.hidden);
            this.requestUpdate();
        }
    } catch (e) {
        this.error = 'Failed to load hidden videos';
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
    if (!motions) return [];
    
    return motions
        .map(motion => ({
            ...motion,
            files: {
                ...motion.files,
                data: motion.files.data.filter(data => {
                    const videoId = `${motion.id}-${data.sample_id}-${data.repetition_id}`;
                    return !this.hidden.has(videoId);
                })
            }
        }))
        .filter(motion => motion.files.data.length > 0);
  }

  private async generateSMPL(motionPath: string, sampleId: number, repId: number, motionId: string) {
    const key = `${motionId}-${sampleId}-${repId}`;
    this.generatingSMPL = { ...this.generatingSMPL, [key]: true };
    
    try {
      const response = await fetch('http://localhost:3000/api/motion/export-smpl', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ motion_path: motionPath })
      });

      const data = await response.json();
      if (data.status === 'success') {
        // Update the motions to include the new SMPL path
        this.motions = this.motions.map(motion => ({
          ...motion,
          files: {
            ...motion.files,
            data: motion.files.data.map((item: any) => {
              if (item.sample_id === sampleId && item.repetition_id === repId) {
                return { ...item, smpl_data: data.smpl_path };
              }
              return item;
            })
          }
        }));
      } else {
        throw new Error(data.message);
      }
    } catch (e) {
      this.error = e instanceof Error ? e.message : 'Failed to generate SMPL data';
    } finally {
      this.generatingSMPL = { ...this.generatingSMPL, [key]: false };
    }
  }

  private async toggleFavorite(motionId: string, sampleId: number, repId: number) {
    const videoId = `${motionId}-${sampleId}-${repId}`;
    try {
        if (this.favorites.has(videoId)) {
            // Remove from favorites
            const response = await fetch('http://localhost:3000/api/motion/favorites', {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ videoId })
            });
            const data = await response.json();
            if (data.status === 'success') {
                this.favorites = new Set(data.favorites);
            }
        } else {
            // Add to favorites
            const response = await fetch('http://localhost:3000/api/motion/favorites', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ videoId })
            });
            const data = await response.json();
            if (data.status === 'success') {
                this.favorites = new Set(data.favorites);
            }
        }
        this.requestUpdate();
    } catch (e) {
        this.error = 'Failed to update favorites';
    }
  }

  private async removeMotion(motionId: string, sampleId: number, repId: number) {
    if (!confirm('Are you sure you want to hide this video?')) return;
    
    try {
        const videoId = `${motionId}-${sampleId}-${repId}`;
        console.log('Attempting to hide video:', videoId);

        const response = await fetch('http://localhost:3000/api/motion/hidden', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ videoId })
        });
        
        const data = await response.json();
        console.log('Server response:', data);

        if (data.status === 'success') {
            this.hidden = new Set(data.hidden);
            this.motions = this.motions.map(motion => ({
                ...motion,
                files: {
                    ...motion.files,
                    data: motion.files.data.filter(d => {
                        const vid = `${motion.id}-${d.sample_id}-${d.repetition_id}`;
                        return !this.hidden.has(vid);
                    })
                }
            })).filter(motion => motion.files.data.length > 0);
            this.requestUpdate();
        } else {
            throw new Error(data.message);
        }
    } catch (e) {
        console.error('Error hiding video:', e);
        this.error = e instanceof Error ? e.message : 'Failed to hide video';
    }
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

        <button 
          class="filter-button favorite ${this.showFavoritesOnly ? 'active' : ''}"
          @click=${() => {
            this.showFavoritesOnly = !this.showFavoritesOnly;
            this.requestUpdate();
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="${this.showFavoritesOnly ? 'currentColor' : 'none'}" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z"/>
          </svg>
          Favorites Only
        </button>
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
                    ${motion.files.data.map(data => {
                      const videoId = `${motion.id}-${data.sample_id}-${data.repetition_id}`;
                      return html`
                        <div class="video-container">
                          <div class="video-header">
                            <span>Sample ${data.sample_id + 1} / Rep ${data.repetition_id + 1}</span>
                            <div class="action-buttons">
                              <button 
                                class="action-button favorite ${this.favorites.has(videoId) ? 'active' : ''}"
                                @click=${() => this.toggleFavorite(motion.id, data.sample_id, data.repetition_id)}
                                title="${this.favorites.has(videoId) ? 'Remove from favorites' : 'Add to favorites'}"
                              >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="${this.favorites.has(videoId) ? 'currentColor' : 'none'}" stroke="currentColor">
                                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z"/>
                                </svg>
                              </button>
                              <button 
                                class="action-button remove"
                                @click=${() => this.removeMotion(motion.id, data.sample_id, data.repetition_id)}
                                title="Remove video"
                              >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                                </svg>
                              </button>
                            </div>
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
                              ${data.smpl_data ? html`
                                <a href=${this.getFileUrl(data.smpl_data)} download class="download-button smpl">
                                  <svg class="download-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                                  </svg>
                                  SMPL Data
                                </a>
                              ` : html`
                                <button 
                                  @click=${() => this.generateSMPL(data.motion_data, data.sample_id, data.repetition_id, motion.id)}
                                  class="download-button smpl ${this.generatingSMPL[`${motion.id}-${data.sample_id}-${data.repetition_id}`] ? 'generating' : ''}"
                                  ?disabled=${this.generatingSMPL[`${motion.id}-${data.sample_id}-${data.repetition_id}`]}
                                >
                                  ${this.generatingSMPL[`${motion.id}-${data.sample_id}-${data.repetition_id}`] ? 'Generating SMPL...' : 'Generate SMPL'}
                                </button>
                              `}
                            </div>
                          </div>
                        </div>
                      `;
                    })}
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
