import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

class MNISTDrawer:
    def __init__(self, network):
        self.network = network
        self.root = tk.Tk()
        self.root.title("Draw a Number (0-9)")
        
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='black')
        self.canvas.pack(pady=10)
        
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.btn_predict = tk.Button(self.root, text="Predict Number", command=self.predict, font=("Arial", 12))
        self.btn_predict.pack(pady=5)
        
        self.btn_clear = tk.Button(self.root, text="Clear Canvas", command=self.clear, font=("Arial", 12))
        self.btn_clear.pack(pady=5)
        
        self.root.mainloop()

    def paint(self, event):
        brush_size = 18 
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        bbox = self.image.getbbox()
        if bbox is None:
            print("Canvas is empty!")
            return
            
        cropped = self.image.crop(bbox)
        
        width, height = cropped.size
        max_dim = max(width, height)
        ratio = 20.0 / max_dim
        new_size = (int(width * ratio), int(height * ratio))
        
        resized = cropped.resize(new_size, resample=Image.Resampling.LANCZOS)
        
        final_image = Image.new("L", (28, 28), 0)
        
        paste_x = (28 - new_size[0]) // 2
        paste_y = (28 - new_size[1]) // 2
        final_image.paste(resized, (paste_x, paste_y))
        
        img_array = np.array(final_image) / 255.0
        
        plt.figure(figsize=(2, 2))
        plt.imshow(img_array, cmap='gray')
        plt.title("Centered (MNIST Style)")
        plt.axis('off')
        plt.show(block=False) 
        
        x = img_array.reshape((784, 1))
        activations = self.network.feedforward(x)
        prediction = np.argmax(activations)
        confidence = float(np.max(activations)) * 100
        
        print(f"Network thinks you drew a: {prediction} (Confidence: {confidence:.2f}%)")