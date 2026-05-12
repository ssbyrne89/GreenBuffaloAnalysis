import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import yfinance as yf
import squarify
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class InteractiveSP500Heatmap:
    def __init__(self):
        self.sp500_data = None
        self.company_rects = {}  # Store rectangle data for hover
        self.hover_annotation = None
        self.sector_mapping = {
            'Information Technology': 'TECHNOLOGY',
            'Health Care': 'HEALTHCARE', 
            'Financials': 'FINANCIAL',
            'Consumer Discretionary': 'CONSUMER CYCLICAL',
            'Communication Services': 'COMMUNICATION SERVICES',
            'Industrials': 'INDUSTRIALS',
            'Consumer Staples': 'CONSUMER DEFENSIVE',
            'Energy': 'ENERGY',
            'Utilities': 'UTILITIES',
            'Real Estate': 'REAL ESTATE',
            'Materials': 'BASIC MATERIALS'
        }
        
    def get_sp500_list(self):
        """Get S&P 500 companies from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            df = tables[0]
            return df[['Symbol', 'Security', 'GICS Sector']]
        except Exception as e:
            print(f"Error fetching S&P 500 list: {e}")
            # Fallback with some major companies
            return pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
                'Security': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'Amazon.com Inc.', 
                           'Tesla Inc.', 'Meta Platforms Inc.', 'NVIDIA Corporation', 'JPMorgan Chase & Co.', 
                           'Johnson & Johnson', 'Visa Inc.'],
                'GICS Sector': ['Information Technology', 'Information Technology', 'Communication Services', 
                              'Consumer Discretionary', 'Consumer Discretionary', 'Communication Services',
                              'Information Technology', 'Financials', 'Health Care', 'Financials']
            })

    def get_market_data(self, symbols, period='2d'):
        """Fetch market data for given symbols"""
        data = {}
        print(f"Fetching data for {len(symbols)} symbols...")

        # Process in batches to avoid rate limiting
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}...")
            try:
                tickers = yf.Tickers(' '.join(batch))
                for symbol in batch:
                    try:
                        ticker = tickers.tickers[symbol]
                        ticker_data = ticker.history(period=period)
                        if len(ticker_data) >= 2:
                            current_price = ticker_data['Close'].iloc[-1]
                            prev_price = ticker_data['Close'].iloc[-2]
                            change_pct = ((current_price - prev_price) / prev_price) * 100

                            # Use actual market cap if available, fall back to price*volume
                            try:
                                market_cap = ticker.info.get('marketCap', None)
                            except Exception:
                                market_cap = None

                            data[symbol] = {
                                'current_price': current_price,
                                'prev_price': prev_price,
                                'change_pct': change_pct,
                                'volume': ticker_data['Volume'].iloc[-1],
                                'market_cap': market_cap
                            }
                    except Exception as e:
                        print(f"Error fetching data for {symbol}: {e}")
                        continue
            except Exception as e:
                print(f"Error with batch starting at {i}: {e}")
                continue

        return data

    def prepare_data(self, limit_per_sector=None):
        """Prepare data for heatmap visualization"""
        # Get S&P 500 companies
        sp500_df = self.get_sp500_list()
        
        # Get market data
        symbols = sp500_df['Symbol'].tolist()
        market_data = self.get_market_data(symbols)
        
        # Combine data
        heatmap_data = []
        for _, row in sp500_df.iterrows():
            symbol = row['Symbol']
            if symbol in market_data:
                sector = self.sector_mapping.get(row['GICS Sector'], row['GICS Sector'])
                md = market_data[symbol]
                heatmap_data.append({
                    'symbol': symbol,
                    'company': row['Security'],
                    'sector': sector,
                    'change_pct': md['change_pct'],
                    'volume': md['volume'],
                    'current_price': md['current_price'],
                    'market_cap': md.get('market_cap')
                })
        
        df = pd.DataFrame(heatmap_data)
        
        # CRITICAL FIX: Check if we have data, if not use sample data
        if len(df) == 0:
            print("\n⚠️  No market data available (network blocked or API down)")
            print("📊 Generating sample data for demonstration...\n")
            return self.get_sample_data()
        
        # Rest of the method stays the same...
        if limit_per_sector:
            print(f"Limiting to top {limit_per_sector} companies per sector by volume...")
            limited_data = []
            for sector in df['sector'].unique():
                sector_data = df[df['sector'] == sector].nlargest(limit_per_sector, 'volume')
                limited_data.append(sector_data)
            df = pd.concat(limited_data, ignore_index=True)
        else:
            print("Including all companies with available data...")
        
        print(f"Total companies with data: {len(df)} out of {len(sp500_df)} S&P 500 companies")
        print(f"Companies per sector:")
        sector_counts = df['sector'].value_counts()
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count} companies")
        
        return df

    def get_sample_data(self):
        """Generate sample S&P 500 data when real data is unavailable"""
        np.random.seed(42)
        
        sectors = {
            'TECHNOLOGY': 75,
            'HEALTHCARE': 65,
            'FINANCIAL': 60,
            'CONSUMER CYCLICAL': 55,
            'COMMUNICATION SERVICES': 25,
            'INDUSTRIALS': 50,
            'CONSUMER DEFENSIVE': 35,
            'ENERGY': 22,
            'UTILITIES': 28,
            'REAL ESTATE': 30,
            'BASIC MATERIALS': 28
        }
        
        data = []
        company_id = 1
        
        for sector, count in sectors.items():
            for i in range(count):
                price = np.random.uniform(20, 500)
                volume = np.random.uniform(1e6, 100e6)
                change_pct = np.random.normal(0, 2)
                
                data.append({
                    'symbol': f'SYM{company_id}',
                    'company': f'Company {company_id}',
                    'sector': sector,
                    'current_price': price,
                    'change_pct': change_pct,
                    'volume': volume
                })
                company_id += 1
        
        print(f"✅ Generated sample data: {len(data)} companies across {len(sectors)} sectors")
        return pd.DataFrame(data)
        
    def create_heatmap(self, data, figsize=(20, 16)):
        """Create unified treemap heatmap with interactive hover functionality"""
        # Create figure with higher DPI for sharper image
        fig, ax = plt.subplots(figsize=figsize, facecolor='black', dpi=150)
        ax.set_facecolor('black')
        
        # Store reference to figure and axis for event handling
        self.fig = fig
        self.ax = ax
        self.company_rects = {}
        
        # Use actual market cap where available, fall back to price * volume
        if 'market_cap' in data.columns and data['market_cap'].notna().any():
            data['market_value'] = data['market_cap'].fillna(data['current_price'] * data['volume'])
        else:
            data['market_value'] = data['current_price'] * data['volume']
        sector_stats = data.groupby('sector').agg({
            'market_value': 'sum',
            'change_pct': 'mean',  # Average change for sector
            'volume': 'sum'
        }).reset_index()
        
        # Sort sectors by market value
        sector_stats = sector_stats.sort_values('market_value', ascending=False)
        
        # Create main sector layout using squarify
        total_area = 100  # Arbitrary total area
        sector_sizes = sector_stats['market_value'].values
        sector_sizes = (sector_sizes / sector_sizes.sum()) * total_area
        
        # Get sector positions for the main treemap
        sector_positions = squarify.squarify(sector_sizes.tolist(), 0, 0, 10, 10)
        
        # Create a mapping of sector to position
        sector_pos_map = {}
        for i, (_, sector_row) in enumerate(sector_stats.iterrows()):
            if i < len(sector_positions):
                sector_pos_map[sector_row['sector']] = sector_positions[i]
        
        # Plot each sector and its companies
        for sector_name, sector_pos in sector_pos_map.items():
            sector_x = sector_pos['x']
            sector_y = sector_pos['y'] 
            sector_w = sector_pos['dx']
            sector_h = sector_pos['dy']
            
            # Get companies in this sector
            sector_companies = data[data['sector'] == sector_name].copy()
            sector_companies = sector_companies.sort_values('market_value', ascending=False)
            
            # Skip if sector area is too small
            if sector_w < 0.2 or sector_h < 0.2:
                continue
            
            # Draw sector border
            sector_border = patches.Rectangle((sector_x, sector_y), sector_w, sector_h,
                                            facecolor='none', edgecolor='white', 
                                            linewidth=2, alpha=0.8)
            ax.add_patch(sector_border)
            
            # Add sector label with better positioning to avoid overlaps
            if sector_w > 1.5 and sector_h > 0.8:
                # Position label at top-left of sector, but lower to avoid main title
                label_x = sector_x + 0.05
                label_y = sector_y + sector_h - 0.25  # Moved down from 0.15
                
                # Only show label if it won't overlap with main title area
                if label_y < 9.0:  # Avoid top area where main title is
                    ax.text(label_x, label_y, sector_name, 
                           fontsize=min(10, sector_w * 1.0), fontweight='bold',  # Reduced font size
                           color='white', ha='left', va='top',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.8))
                else:
                    # For sectors at the top, position label differently
                    label_y = sector_y + 0.1  # Position at bottom of sector instead
                    ax.text(label_x, label_y, sector_name, 
                           fontsize=min(10, sector_w * 1.0), fontweight='bold', 
                           color='white', ha='left', va='bottom',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.8))
            
            # Create company layout within sector
            if len(sector_companies) > 0:
                company_sizes = sector_companies['market_value'].values
                
                # Use more of the available sector space (95% instead of 85%)
                company_area = sector_w * sector_h * 0.95  
                normalized_company_sizes = (company_sizes / company_sizes.sum()) * company_area
                
                # Calculate better margins based on sector size
                margin_x = max(0.02, sector_w * 0.01)
                margin_y = max(0.02, sector_h * 0.01)
                
                # Get company positions within the sector with optimized space usage
                try:
                    company_positions = squarify.squarify(
                        normalized_company_sizes.tolist(), 
                        sector_x + margin_x, 
                        sector_y + margin_y, 
                        sector_w - 2 * margin_x, 
                        sector_h - 2 * margin_y
                    )
                except:
                    # Fallback for edge cases
                    company_positions = []
                
                # Plot companies with improved space utilization
                for i, (_, company) in enumerate(sector_companies.iterrows()):
                    if i < len(company_positions):
                        comp_pos = company_positions[i]
                        x, y, w, h = comp_pos['x'], comp_pos['y'], comp_pos['dx'], comp_pos['dy']
                        
                        # Reduce minimum size threshold to use more space
                        if w < 0.02 or h < 0.02:
                            continue
                        
                        # Color based on percentage change
                        change = company['change_pct']
                        
                        # Enhanced color mapping
                        if change >= 0:
                            # Green for positive changes
                            intensity = min(abs(change) / 3.0, 1.0)
                            color = plt.cm.Greens(0.4 + intensity * 0.6)
                        else:
                            # Red for negative changes
                            intensity = min(abs(change) / 3.0, 1.0)
                            color = plt.cm.Reds(0.4 + intensity * 0.6)
                        
                        # Draw company rectangle with minimal gaps
                        rect = patches.Rectangle((x, y), w, h, 
                                               facecolor=color, 
                                               edgecolor='white', 
                                               linewidth=0.3, alpha=0.95)  # Thinner borders
                        ax.add_patch(rect)
                        
                        # Store rectangle data for hover functionality
                        rect_id = f"{company['symbol']}_{i}"
                        self.company_rects[rect_id] = {
                            'rect': rect,
                            'bounds': (x, y, x+w, y+h),
                            'data': {
                                'symbol': company['symbol'],
                                'company': company['company'],
                                'sector': sector_name,
                                'price': company['current_price'],
                                'change_pct': company['change_pct'],
                                'volume': company['volume'],
                                'market_value': company['market_value']
                            }
                        }
                        
                        # Adaptive text sizing with better thresholds
                        min_text_size = 0.12
                        if w > min_text_size and h > min_text_size:
                            # Company symbol with better scaling
                            symbol_size = min(10, max(4, min(w * 10, h * 12)))
                            ax.text(x + w/2, y + h/2 + h*0.08, company['symbol'], 
                                   fontsize=symbol_size, fontweight='bold', color='white', 
                                   ha='center', va='center')
                            
                            # Percentage change (only if there's enough space)
                            if h > 0.25:  # Reduced from 0.4
                                pct_size = min(7, max(3, min(w * 8, h * 8)))
                                ax.text(x + w/2, y + h/2 - h*0.08, f"{change:+.1f}%", 
                                       fontsize=pct_size, color='white', 
                                       ha='center', va='center', alpha=0.9)
                        
                        # For smaller rectangles, just show symbol
                        elif w > 0.06 and h > 0.06:
                            symbol_size = max(3, min(w * 8, h * 8))
                            ax.text(x + w/2, y + h/2, company['symbol'], 
                                   fontsize=symbol_size, fontweight='bold', color='white', 
                                   ha='center', va='center')
        
        # Set plot limits
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with proper spacing
        fig.suptitle('S&P 500 Interactive Market Treemap - Hover for Details', 
                    fontsize=16, fontweight='bold', color='white', y=0.975)  # Even smaller font, specific position
        
        # Add subtitle with much more separation
        subtitle = f"Daily price changes • Sector size = total market value • {datetime.now().strftime('%B %d, %Y')}"
        fig.text(0.5, 0.925, subtitle, fontsize=8, color='#cccccc',  # Much lower (0.05 gap), smaller font
                ha='center', va='center')
        
        # Add legend and sector list
        self._add_legend(ax, fig)
        self._add_sector_legend(fig, sector_stats)
        
        # Connect mouse events for interactivity
        self._setup_interactivity()
        
        # Adjust layout with more space for wider legend
        plt.subplots_adjust(left=0.02, right=0.81, top=0.90, bottom=0.02)
        
        return fig

    def _setup_interactivity(self):
        """Setup mouse hover interactivity"""
        # Connect mouse motion event
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Initialize hover annotation
        self.hover_annotation = None

    def _on_mouse_move(self, event):
        """Handle mouse movement for hover effects"""
        if event.inaxes != self.ax:
            if self.hover_annotation:
                self.hover_annotation.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        
        # Check if mouse is over any company rectangle
        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            return
        
        hovered_company = None
        for rect_id, rect_data in self.company_rects.items():
            x_min, y_min, x_max, y_max = rect_data['bounds']
            if x_min <= mouse_x <= x_max and y_min <= mouse_y <= y_max:
                hovered_company = rect_data
                break
        
        if hovered_company:
            self._show_hover_info(hovered_company, mouse_x, mouse_y)
        else:
            if self.hover_annotation:
                self.hover_annotation.set_visible(False)
                self.fig.canvas.draw_idle()

    def _show_hover_info(self, company_data, x, y):
        """Display hover information for a company with smart positioning"""
        data = company_data['data']
        
        # Format the information
        info_text = (
            f"{data['symbol']}\n"
            f"{data['company'][:25]}{'...' if len(data['company']) > 25 else ''}\n"
            f"Sector: {data['sector']}\n"
            f"Price: ${data['price']:.2f}\n"
            f"Change: {data['change_pct']:+.2f}%\n"
            f"Volume: {data['volume']:,.0f}\n"
            f"Market Value: ${data['market_value']:,.0f}"
        )
        
        # Remove existing annotation
        if self.hover_annotation:
            self.hover_annotation.remove()
        
        # Get axis limits for boundary checking
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        # Calculate smart positioning
        # Default offset
        offset_x, offset_y = 20, 20
        h_align = 'left'
        v_align = 'bottom'
        
        # Adjust horizontal positioning
        if x > (x_max - x_min) * 0.7 + x_min:  # Near right edge
            offset_x = -20
            h_align = 'right'
        
        # Adjust vertical positioning  
        if y > (y_max - y_min) * 0.8 + y_min:  # Near top edge
            offset_y = -20
            v_align = 'top'
        
        # For corners, adjust both
        if x > (x_max - x_min) * 0.7 + x_min and y > (y_max - y_min) * 0.8 + y_min:
            offset_x, offset_y = -20, -20
            h_align, v_align = 'right', 'top'
        elif x < (x_max - x_min) * 0.3 + x_min and y > (y_max - y_min) * 0.8 + y_min:
            offset_x, offset_y = 20, -20
            h_align, v_align = 'left', 'top'
        elif x > (x_max - x_min) * 0.7 + x_min and y < (y_max - y_min) * 0.2 + y_min:
            offset_x, offset_y = -20, 20
            h_align, v_align = 'right', 'bottom'
        
        # Create new annotation with smart positioning
        self.hover_annotation = self.ax.annotate(
            info_text,
            xy=(x, y),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.95, 
                     edgecolor='white', linewidth=1),
            fontsize=9,
            color='white',
            ha=h_align,
            va=v_align,
            fontfamily='monospace',
            zorder=1000  # Ensure it appears on top
        )
        
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse clicks for additional interactivity"""
        if event.inaxes != self.ax:
            return
        
        # Check if click is on any company rectangle
        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            return
        
        for rect_id, rect_data in self.company_rects.items():
            x_min, y_min, x_max, y_max = rect_data['bounds']
            if x_min <= mouse_x <= x_max and y_min <= mouse_y <= y_max:
                data = rect_data['data']
                print(f"\n📊 Clicked on {data['symbol']} ({data['company']})")
                print(f"💰 Current Price: ${data['price']:.2f}")
                print(f"📈 Daily Change: {data['change_pct']:+.2f}%")
                print(f"📊 Volume: {data['volume']:,.0f}")
                print(f"🏢 Sector: {data['sector']}")
                print(f"💵 Market Value: ${data['market_value']:,.0f}")
                break

    def _add_legend(self, ax, fig):
        """Add color legend for unified treemap"""
        # Create legend showing color scale - moved much higher to avoid cutoff
        legend_x = 0.02
        legend_y = 0.15  # Raised significantly from 0.06
        legend_width = 0.15
        legend_height = 0.03
        
        # Create color gradient
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        
        # Create legend axis
        legend_ax = fig.add_axes([legend_x, legend_y, legend_width, legend_height])
        legend_ax.imshow(gradient, aspect='auto', cmap='RdYlGn', extent=[-3, 3, 0, 1])
        legend_ax.set_xlim(-3, 3)
        legend_ax.set_ylim(0, 1)
        
        # Add labels
        legend_ax.set_xticks([-3, -1.5, 0, 1.5, 3])
        legend_ax.set_xticklabels(['-3%', '-1.5%', '0%', '+1.5%', '+3%'], 
                                 color='white', fontsize=8)
        legend_ax.set_yticks([])
        legend_ax.set_xlabel('Daily Price Change', color='white', fontsize=9)
        
        # Style the legend
        legend_ax.tick_params(colors='white', labelsize=8)
        for spine in legend_ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
            
        # Add market cap info well below the legend to avoid all overlaps
        info_text = "Rectangle size = Market Value (Price × Volume)"
        fig.text(0.02, 0.04, info_text, fontsize=8, color='#cccccc')  # Much lower and smaller

    def _add_sector_legend(self, fig, sector_stats):
        """Add sector list ordered by market volume on the right side"""
        # Create sector legend area with more width
        legend_x = 0.82
        legend_y = 0.15
        legend_width = 0.17  # Increased from 0.15
        legend_height = 0.75
        
        # Create legend axis
        sector_ax = fig.add_axes([legend_x, legend_y, legend_width, legend_height])
        sector_ax.set_facecolor('black')
        sector_ax.axis('off')
        
        # Sort sectors by market value (already sorted in sector_stats)
        y_positions = np.linspace(0.85, 0.10, len(sector_stats))  # More space between entries
        
        # Add title with shorter text and better positioning
        sector_ax.text(0.01, 0.95, 'SECTORS', 
                      fontsize=11, fontweight='bold', color='white', 
                      ha='left', va='top', transform=sector_ax.transAxes)
        
        # Add each sector with its market value
        for i, (_, sector_row) in enumerate(sector_stats.iterrows()):
            y_pos = y_positions[i]
            sector_name = sector_row['sector']
            market_val = sector_row['market_value']
            avg_change = sector_row['change_pct']
            
            # Format market value
            if market_val >= 1e12:
                val_text = f"${market_val/1e12:.1f}T"
            elif market_val >= 1e9:
                val_text = f"${market_val/1e9:.1f}B"
            elif market_val >= 1e6:
                val_text = f"${market_val/1e6:.0f}M"
            else:
                val_text = f"${market_val:.0f}"
            
            # Color based on average sector performance
            if avg_change >= 0:
                color = '#90EE90'  # Light green
            else:
                color = '#FFB6C1'  # Light red
            
            # Sector name (truncated if too long)
            display_name = sector_name if len(sector_name) <= 16 else sector_name[:13] + "..."
            
            # Add sector info with better horizontal spacing
            sector_ax.text(0.01, y_pos, f"{i+1}.", 
                          fontsize=8, color='#cccccc', 
                          ha='left', va='center', transform=sector_ax.transAxes)
            
            # Sector name on its own line with more space from number
            sector_ax.text(0.12, y_pos + 0.015, display_name, 
                          fontsize=8, fontweight='bold', color='white', 
                          ha='left', va='center', transform=sector_ax.transAxes)
            
            # Market value below sector name with same left alignment
            sector_ax.text(0.12, y_pos - 0.015, val_text, 
                          fontsize=7, color=color, 
                          ha='left', va='center', transform=sector_ax.transAxes)
            
            # Percentage aligned with market value line, positioned far right
            sector_ax.text(0.95, y_pos - 0.015, f"{avg_change:+.1f}%", 
                          fontsize=7, color=color, fontweight='bold',
                          ha='right', va='center', transform=sector_ax.transAxes)
        
        # Add footer info
        sector_ax.text(0.02, 0.02, "Market Value = Price × Volume", 
                      fontsize=7, color='#888888', style='italic',
                      ha='left', va='bottom', transform=sector_ax.transAxes)

    def generate_heatmap(self, save_path=None, show=True):
        """Main function to generate the heatmap"""
        print("Preparing S&P 500 data...")
        data = self.prepare_data()
        
        print(f"Creating heatmap with {len(data)} companies across {data['sector'].nunique()} sectors...")
        fig = self.create_heatmap(data)
        
        if save_path:
            # Save with high DPI and quality settings
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none', 
                       pad_inches=0.1, format='png')
            print(f"High-resolution heatmap saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig

# Usage example
if __name__ == "__main__":
    # Create interactive heatmap generator
    heatmap = InteractiveSP500Heatmap()
    
    # Generate and display interactive heatmap
    fig = heatmap.generate_heatmap(save_path="sp500_interactive_heatmap.png", show=True)
    
    print("\n🎯 Interactive S&P 500 Heatmap generated successfully!")
    print("✨ Features:")
    print("  • Hover over any company square to see detailed information")
    print("  • Click on any company square to print details to console")
    print("  • Green = Positive price change")
    print("  • Red = Negative price change")
    print("  • Rectangle size = Market value (Price × Volume)")
    print("\n💡 Try hovering and clicking on different companies!")
