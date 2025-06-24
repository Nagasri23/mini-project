import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analyzer")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize model variables
        self.model = None
        self.vectorizer = None
        self.df = None
        
        # Create container frame for all pages
        self.container = tk.Frame(self.root, bg='#f0f0f0')
        self.container.pack(fill='both', expand=True)
        
        # Dictionary to hold all frames
        self.frames = {}
        
        # Create all pages but don't show them yet
        self.create_header()
        self.create_welcome_page()
        self.create_load_page()
        self.create_train_page()
        self.create_test_page()
        self.create_stats_page()
        
        # Show the welcome page initially
        self.show_frame("WelcomePage")
        
    def create_header(self):
        # Header frame that stays visible always
        self.header_frame = tk.Frame(self.root, bg='#4a6baf')
        self.header_frame.pack(fill='x', padx=10, pady=10)
        
        self.header_label = tk.Label(
            self.header_frame, 
            text="Sentiment Analysis Tool", 
            font=('Helvetica', 18, 'bold'), 
            fg='white', 
            bg='#4a6baf'
        )
        self.header_label.pack(pady=10)
        
        # Navigation buttons
        nav_frame = tk.Frame(self.header_frame, bg='#4a6baf')
        nav_frame.pack(pady=5)
        
        buttons = [
            ("Welcome", "WelcomePage"),
            ("Load Data", "LoadPage"),
            ("Train Model", "TrainPage"),
            ("Test Sentence", "TestPage"),
            ("Statistics", "StatsPage")
        ]
        
        for text, page in buttons:
            btn = tk.Button(
                nav_frame, 
                text=text, 
                command=lambda p=page: self.show_frame(p),
                bg='#4a6baf',  # Blue background
                fg='white',     # White text
                font=('Helvetica', 10, 'bold'),
                relief='raised',
                bd=2,
                padx=10,
                pady=5,
                activebackground='#3a5a9f',  # Slightly darker blue when pressed
                activeforeground='white'
            )
            btn.pack(side='left', padx=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root, 
            text="Ready", 
            bd=1, 
            relief='sunken', 
            anchor='w',
            bg='#e0e0e0'
        )
        self.status_bar.pack(fill='x', padx=10, pady=5)
    
    def create_welcome_page(self):
        frame = tk.Frame(self.container, bg='#f0f0f0')
        self.frames["WelcomePage"] = frame
        
        welcome_label = tk.Label(
            frame, 
            text="Welcome to Sentiment Analyzer!\n\nThis tool can analyze text sentiment using machine learning.\n\nPlease use the navigation buttons above to get started.",
            font=('Helvetica', 14), 
            bg='#f0f0f0',
            justify='center'
        )
        welcome_label.pack(pady=50)
        
        # Add some instructions
        instructions = tk.Label(
            frame,
            text="How to use:\n1. Load your dataset (CSV with 'text' and 'label' columns)\n2. Train the model\n3. Test sentences\n4. View statistics",
            font=('Helvetica', 12),
            bg='#f0f0f0',
            justify='left'
        )
        instructions.pack(pady=20)
        
        frame.pack_propagate(False)
    
    def create_load_page(self):
        frame = tk.Frame(self.container, bg='#f0f0f0')
        self.frames["LoadPage"] = frame
        
        # Title
        title_label = tk.Label(
            frame, 
            text="Load Dataset", 
            font=('Helvetica', 16, 'bold'), 
            bg='#f0f0f0'
        )
        title_label.pack(pady=20)
        
        # Load button with blue background and white text
        load_btn = tk.Button(
            frame, 
            text="Select CSV File", 
            command=self.load_dataset,
            bg='#4a6baf',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            relief='raised',
            bd=2,
            padx=15,
            pady=5,
            activebackground='#3a5a9f',
            activeforeground='white'
        )
        load_btn.pack(pady=10)
        
        # Display area
        self.load_display = tk.Label(
            frame, 
            text="No dataset loaded", 
            font=('Helvetica', 12), 
            bg='white',
            relief='groove',
            width=80,
            height=10,
            wraplength=700,
            justify='left'
        )
        self.load_display.pack(pady=20, padx=20)
        
        frame.pack_propagate(False)
    
    def create_train_page(self):
        frame = tk.Frame(self.container, bg='#f0f0f0')
        self.frames["TrainPage"] = frame
        
        # Title
        title_label = tk.Label(
            frame, 
            text="Train Model", 
            font=('Helvetica', 16, 'bold'), 
            bg='#f0f0f0'
        )
        title_label.pack(pady=20)
        
        # Train button with blue background and white text
        train_btn = tk.Button(
            frame, 
            text="Train Model", 
            command=self.train_model,
            bg='#4a6baf',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            relief='raised',
            bd=2,
            padx=15,
            pady=5,
            activebackground='#3a5a9f',
            activeforeground='white'
        )
        train_btn.pack(pady=10)
        
        # Display area
        self.train_display = tk.Label(
            frame, 
            text="Model not trained yet", 
            font=('Helvetica', 12), 
            bg='white',
            relief='groove',
            width=80,
            height=15,
            wraplength=700,
            justify='left'
        )
        self.train_display.pack(pady=20, padx=20)
        
        frame.pack_propagate(False)
    
    def create_test_page(self):
        frame = tk.Frame(self.container, bg='#f0f0f0')
        self.frames["TestPage"] = frame
        
        # Title
        title_label = tk.Label(
            frame, 
            text="Test Sentence", 
            font=('Helvetica', 16, 'bold'), 
            bg='#f0f0f0'
        )
        title_label.pack(pady=20)
        
        # Input area
        input_frame = tk.Frame(frame, bg='#f0f0f0')
        input_frame.pack(pady=10)
        
        tk.Label(
            input_frame, 
            text="Enter a sentence:", 
            font=('Helvetica', 12), 
            bg='#f0f0f0'
        ).pack(side='left')
        
        self.sentence_entry = tk.Entry(
            input_frame, 
            font=('Helvetica', 12), 
            width=50
        )
        self.sentence_entry.pack(side='left', padx=10)
        
        # Analyze button with blue background and white text
        analyze_btn = tk.Button(
            frame, 
            text="Analyze", 
            command=self.analyze_sentence,
            bg='#4a6baf',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            relief='raised',
            bd=2,
            padx=15,
            pady=5,
            activebackground='#3a5a9f',
            activeforeground='white'
        )
        analyze_btn.pack(pady=10)
        
        # Result display
        self.result_display = tk.Label(
            frame, 
            text="Results will appear here", 
            font=('Helvetica', 12), 
            bg='white',
            relief='groove',
            width=80,
            height=10,
            wraplength=700,
            justify='left'
        )
        self.result_display.pack(pady=20, padx=20)
        
        frame.pack_propagate(False)
    
    def create_stats_page(self):
        frame = tk.Frame(self.container, bg='#f0f0f0')
        self.frames["StatsPage"] = frame
        
        # Title
        title_label = tk.Label(
            frame, 
            text="Dataset Statistics", 
            font=('Helvetica', 16, 'bold'), 
            bg='#f0f0f0'
        )
        title_label.pack(pady=20)
        
        # Stats display
        self.stats_display = tk.Label(
            frame, 
            text="No dataset loaded", 
            font=('Helvetica', 12), 
            bg='white',
            relief='groove',
            width=80,
            height=15,
            wraplength=700,
            justify='left'
        )
        self.stats_display.pack(pady=20, padx=20)
        
        # Show plot button with blue background and white text
        plot_btn = tk.Button(
            frame, 
            text="Show Distribution Plot", 
            command=self.show_distribution_plot,
            bg='#4a6baf',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            relief='raised',
            bd=2,
            padx=15,
            pady=5,
            activebackground='#3a5a9f',
            activeforeground='white'
        )
        plot_btn.pack(pady=10)
        
        frame.pack_propagate(False)
    
    def show_frame(self, page_name):
        # Hide all frames
        for frame in self.frames.values():
            frame.pack_forget()
        
        # Show the requested frame
        frame = self.frames[page_name]
        frame.pack(fill='both', expand=True)
        
        # Update stats if showing stats page
        if page_name == "StatsPage" and self.df is not None:
            self.update_stats_display()
    
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset", 
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.df = self.df[['text', 'label']]
                self.df.dropna(subset=['text', 'label'], inplace=True)
                self.df['label'] = self.df['label'].str.lower()
                
                # Update display
                self.load_display.config(
                    text=f"Dataset loaded successfully!\n\nRecords: {len(self.df)}\n\nLabel distribution:\n{self.df['label'].value_counts().to_string()}"
                )
                
                self.status_bar.config(text=f"Dataset loaded successfully with {len(self.df)} records")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
                self.status_bar.config(text="Error loading dataset")
    
    def train_model(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            self.status_bar.config(text="No dataset loaded")
            return
            
        try:
            # Vectorization
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            X = self.vectorizer.fit_transform(self.df['text'])
            y = self.df['label']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestClassifier(random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Show results
            self.train_display.config(
                text=f"Model trained successfully!\n\nAccuracy: {accuracy:.2f}\n\nClassification Report:\n{report}"
            )
            
            self.status_bar.config(text="Model trained successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_bar.config(text="Error training model")
    
    def analyze_sentence(self):
        if self.model is None or self.vectorizer is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            self.status_bar.config(text="Model not trained")
            return
            
        user_sentence = self.sentence_entry.get()
        
        if user_sentence:
            try:
                input_vec = self.vectorizer.transform([user_sentence])
                prediction = self.model.predict(input_vec)[0]
                
                # Color the prediction result
                color = 'green' if prediction == 'happy' else 'red' if prediction == 'sad' else 'blue'
                
                self.result_display.config(
                    text=f"Sentence: {user_sentence}\n\nPredicted sentiment: {prediction}",
                    fg=color
                )
                
                self.status_bar.config(text="Sentence analyzed successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to analyze sentence: {str(e)}")
                self.status_bar.config(text="Error analyzing sentence")
        else:
            messagebox.showwarning("Warning", "Please enter a sentence first!")
            self.status_bar.config(text="No sentence entered")
    
    def update_stats_display(self):
        if self.df is not None:
            stats_text = f"""
            Dataset Information:
            
            Total records: {len(self.df)}
            Label distribution:
            {self.df['label'].value_counts().to_string()}
            
            Sample records:
            {self.df.head(3).to_string()}
            """
            
            self.stats_display.config(text=stats_text)
    
    def show_distribution_plot(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
            
        try:
            # Create a new window for the plot
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Label Distribution")
            plot_window.geometry("600x500")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.countplot(x='label', data=self.df, palette='viridis', ax=ax)
            ax.set_title('Label Distribution in Dataset')
            ax.set_xlabel('Sentiment Label')
            ax.set_ylabel('Count')
            plt.tight_layout()
            
            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Add a close button with blue background and white text
            close_btn = tk.Button(
                plot_window, 
                text="Close", 
                command=plot_window.destroy,
                bg='#4a6baf',
                fg='white',
                font=('Helvetica', 10, 'bold'),
                relief='raised',
                bd=2,
                padx=15,
                pady=5,
                activebackground='#3a5a9f',
                activeforeground='white'
            )
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show plot: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()