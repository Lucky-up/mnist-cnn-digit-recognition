import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import torchvision.transforms as transforms
import tkinter as tk

# 【必须和训练代码网络完全一致】
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

# 加载模型
device = torch.device("cpu")
model = BetterCNN().to(device)
# 加载你训练好的权重
model.load_state_dict(torch.load("models/better_cnn_mnist.pth", map_location=device))
model.eval()

# 数据变换（和训练保持一致）
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 图片预处理：适配MNIST风格（去白边、居中、模糊）
def mnist_style_transform(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    bbox = img.getbbox()
    if not bbox:
        return Image.new("L", (28, 28), 0)
    img = img.crop(bbox)
    img = img.resize((20, 20), Image.BILINEAR)
    canvas = Image.new("L", (28, 28), 0)
    offset = (28 - 20) // 2
    canvas.paste(img, (offset, offset))
    canvas = ImageOps.invert(canvas)
    return canvas

# 手写GUI界面（修复布局报错）
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("增强版手写数字识别")
        self.root.geometry("320x400")

        self.canvas_size = 300
        self.canvas = tk.Canvas(root, bg="white", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(pady=10)

        self.img = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.canvas.bind("<B1-Motion>", self.on_draw)

        # 按钮区域（修正：column/padx 只写在 grid 里）
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        btn1 = tk.Button(btn_frame, text="识别", command=self.predict, width=10)
        btn1.grid(row=0, column=0, padx=15)
        btn2 = tk.Button(btn_frame, text="清空", command=self.clear, width=10)
        btn2.grid(row=0, column=1, padx=15)

        self.result = tk.Label(root, text="请书写 0~9 单个数字", font=("Arial", 18))
        self.result.pack(pady=15)

        self.last_x = None
        self.last_y = None
        self.pen_width = 8

    def on_draw(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill="black", width=self.pen_width, capstyle=tk.ROUND
            )
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=self.pen_width)
        self.last_x, self.last_y = x, y

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.last_x = self.last_y = None
        self.result.config(text="已清空，请重新书写")

    def predict(self):
        proc_img = mnist_style_transform(self.img)
        tensor = base_transform(proc_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred = torch.argmax(out, dim=1).item()
        self.result.config(text=f"识别结果：{pred}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
