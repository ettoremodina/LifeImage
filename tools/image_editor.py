"""Summary: Interactive image editor for creating target images for fitness evaluation."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import argparse
from pathlib import Path


class ImageEditor:
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.cell_size = max(8, 512 // grid_size)
        self.canvas_size = self.cell_size * grid_size
        
        # Image data: 0 = black, 1 = white
        self.image_data = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        self.root = tk.Tk()
        self.root.title(f"Target Image Editor ({grid_size}x{grid_size})")
        self.root.resizable(False, False)
        
        self.setup_ui()
        self.drawing = False
        self.erase_mode = False
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas
        self.canvas = tk.Canvas(
            main_frame, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg='black',
            bd=2,
            relief='sunken'
        )
        self.canvas.grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.canvas.bind("<Button-3>", self.start_erase)
        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.bind("<ButtonRelease-3>", self.stop_erase)
        
        # Controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E))
        
        ttk.Button(controls_frame, text="Clear", command=self.clear_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(controls_frame, text="Save", command=self.save_image).grid(row=0, column=1, padx=5)
        ttk.Button(controls_frame, text="Load", command=self.load_image).grid(row=0, column=2, padx=5)
        ttk.Button(controls_frame, text="Random", command=self.random_pattern).grid(row=0, column=3, padx=(5, 0))
        
        # Instructions
        instructions = ttk.Label(
            main_frame, 
            text="Left click/drag: Draw white pixels | Right click/drag: Erase (black pixels)",
            font=('Arial', 9)
        )
        instructions.grid(row=2, column=0, columnspan=4, pady=(10, 0))
        
        self.update_canvas()
    
    def pixel_to_grid(self, x: int, y: int) -> tuple[int, int]:
        """Convert canvas coordinates to grid coordinates."""
        grid_x = min(self.grid_size - 1, max(0, x // self.cell_size))
        grid_y = min(self.grid_size - 1, max(0, y // self.cell_size))
        return grid_x, grid_y
    
    def start_draw(self, event):
        self.drawing = True
        self.draw(event)
    
    def draw(self, event):
        if not self.drawing:
            return
        grid_x, grid_y = self.pixel_to_grid(event.x, event.y)
        self.image_data[grid_y, grid_x] = 1.0
        self.update_pixel(grid_x, grid_y)
    
    def stop_draw(self, event):
        self.drawing = False
    
    def start_erase(self, event):
        self.erase_mode = True
        self.erase(event)
    
    def erase(self, event):
        if not self.erase_mode:
            return
        grid_x, grid_y = self.pixel_to_grid(event.x, event.y)
        self.image_data[grid_y, grid_x] = 0.0
        self.update_pixel(grid_x, grid_y)
    
    def stop_erase(self, event):
        self.erase_mode = False
    
    def update_pixel(self, grid_x: int, grid_y: int):
        """Update a single pixel on the canvas."""
        x1 = grid_x * self.cell_size
        y1 = grid_y * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        
        color = 'white' if self.image_data[grid_y, grid_x] > 0.5 else 'black'
        
        # Remove existing rectangle if any
        tag = f"pixel_{grid_x}_{grid_y}"
        self.canvas.delete(tag)
        
        # Draw new rectangle
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray', tags=tag)
    
    def update_canvas(self):
        """Redraw the entire canvas."""
        self.canvas.delete("all")
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.update_pixel(x, y)
    
    def clear_image(self):
        """Clear the entire image."""
        self.image_data.fill(0.0)
        self.update_canvas()
    
    def save_image(self):
        """Save the current image."""
        filename = filedialog.asksaveasfilename(
            title="Save Target Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialdir="assets/target"
        )
        
        if filename:
            # Convert to RGB image
            rgb_data = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            rgb_data[:, :, 0] = self.image_data * 255  # R
            rgb_data[:, :, 1] = self.image_data * 255  # G  
            rgb_data[:, :, 2] = self.image_data * 255  # B
            
            img = Image.fromarray(rgb_data, 'RGB')
            img.save(filename)
            messagebox.showinfo("Saved", f"Image saved to {filename}")
    
    def load_image(self):
        """Load an existing image."""
        filename = filedialog.askopenfilename(
            title="Load Target Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")],
            initialdir="assets/target"
        )
        
        if filename:
            try:
                img = Image.open(filename).convert("RGB").resize((self.grid_size, self.grid_size))
                img_array = np.asarray(img, dtype=np.float32) / 255.0
                # Convert to grayscale and threshold
                self.image_data = (img_array.mean(axis=2) > 0.5).astype(np.float32)
                self.update_canvas()
                messagebox.showinfo("Loaded", f"Image loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def random_pattern(self):
        """Generate a random pattern."""
        import random
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.image_data[y, x] = 1.0 if random.random() > 0.7 else 0.0
        self.update_canvas()
    
    def run(self):
        """Start the image editor."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Interactive target image editor")
    parser.add_argument("--size", type=int, default=64, help="Grid size (default: 64)")
    args = parser.parse_args()
    
    # Ensure assets/target directory exists
    target_dir = Path("assets/target")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    editor = ImageEditor(args.size)
    editor.run()


if __name__ == "__main__":
    main()
