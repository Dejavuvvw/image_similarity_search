# ---------- 后端 Python (main.py) ----------
import os
import h5py
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
from tqdm import tqdm
import json
from fastapi.responses import JSONResponse
import logging
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("app.log")  # 同时输出到文件
    ]
)

app = FastAPI()

# 配置静态文件目录
app.mount("/static", StaticFiles(directory="images"), name="static")

# 初始化ResNet-50模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).eval()
model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 特征存储配置
FEATURES_PATH = "features.h5"
IMAGE_DIR = "images"
TRASH_DIR = "trash"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TRASH_DIR, exist_ok=True)
# 新增配置项
TRASH_FILE = "trash_list.json"

# 加载废弃列表
def load_trash():
    try:
        with open(TRASH_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

class ImageDataset(data.Dataset):
    """批量处理数据集"""
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = [f for f in os.listdir(img_dir) 
                       if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        filename = self.img_list[idx]
        try:
            img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
            return preprocess(img), filename
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None, None

# 保存废弃列表
def save_trash(trash_list):
    with open(TRASH_FILE, 'w') as f:
        json.dump(trash_list, f)

def extract_features(img_path):
    """提取图片特征向量"""
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_t)
    return features.cpu().numpy().flatten()

def build_feature_db():
    """批量构建特征数据库"""
    logger.info("开始构建特征数据库...")
    dataset = ImageDataset(IMAGE_DIR)
    
    # 创建可扩展的HDF5数据集
    with h5py.File(FEATURES_PATH, 'w') as hf:
        features_ds = hf.create_dataset(
            "features", 
            shape=(0, 2048), 
            maxshape=(None, 2048),  # 可扩展维度
            dtype=np.float32
        )
        filenames_ds = hf.create_dataset(
            "filenames",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        
        # 数据加载配置
        loader = data.DataLoader(
            dataset,
            batch_size=32,  # 根据GPU内存调整
            shuffle=False,
            num_workers=16,   # 并行加载进程数
            collate_fn=lambda x: [item for item in x if item[0] is not None]
        )
        
        # 批量处理
        with torch.no_grad():
            for batch in tqdm(loader, desc="Processing batches"):
                if not batch:
                    continue
                
                imgs, filenames = zip(*batch)
                img_tensors = torch.stack(imgs).to(device)
                
                # 批量特征提取
                batch_features = model(img_tensors)
                batch_features = batch_features.squeeze().cpu().numpy()
                
                # 扩展HDF5数据集
                current_size = features_ds.shape[0]
                new_size = current_size + len(batch_features)
                
                features_ds.resize(new_size, axis=0)
                features_ds[current_size:new_size] = batch_features
                
                filenames_ds.resize(new_size, axis=0)
                filenames_ds[current_size:new_size] = filenames
    
    with h5py.File(FEATURES_PATH, 'r') as hf:
        assert len(hf['features']) == len(hf['filenames']), "数据不一致"
        print(f"成功处理 {len(hf['features'])} 张图片")
        print("样例数据：", hf['features'][0][:10])


@app.on_event("startup")
async def startup_event():
    """启动时加载/构建特征库"""
    if not os.path.exists(FEATURES_PATH):
        build_feature_db()


# 新增API端点
@app.post("/trash/add")
async def add_to_trash(request: Request):
    data = await request.json()
    filename = data.get('filename')
    if not filename:
        raise HTTPException(400, "缺少文件名参数")
    # 提取纯文件名（防止前端传递完整URL）
    pure_filename = os.path.basename(filename)

    trash_list = load_trash()
    if filename not in trash_list: 
        trash_list.append(filename)
        save_trash(trash_list)
        logger.info(f"已添加文件到废弃区: {pure_filename}")
    return JSONResponse({"status": "success", "filename": pure_filename})


@app.post("/trash/remove")
async def remove_from_trash(request: Request):
    data = await request.json()
    filename = data.get('filename')
    
    if not filename:
        raise HTTPException(400, "缺少文件名参数")
    
    trash_list = load_trash()
    if filename in trash_list:
        trash_list.remove(filename)
        save_trash(trash_list)
        logger.info(f"已从废弃区移除文件: {filename}")
    
    return JSONResponse({"status": "success"})


@app.get("/trash/list")
async def get_trash():
    return JSONResponse({"files": load_trash()})

@app.post("/trash/confirm")
async def confirm_trash():
    trash_list = load_trash()
    deleted_files = []
    image_dir = os.path.abspath(IMAGE_DIR)
    trash_dir = os.path.abspath(TRASH_DIR)

    for filename in trash_list:
        try:
            file_path = os.path.join(image_dir, filename)
            
            if not os.path.abspath(file_path).startswith(image_dir):
                logging.error(f"非法路径: {file_path}")
                continue

            if os.path.exists(file_path):
                shutil.move(file_path, os.path.join(trash_dir, filename))
                deleted_files.append(filename)
                logging.info(f"已删除文件: {file_path}")
            else:
                logging.warning(f"文件不存在: {file_path}")
        except Exception as e:
            logging.error(f"删除失败 {filename}: {str(e)}")

    # 修复方法：先将数据读取到内存中，然后关闭文件，再创建新文件
    if os.path.exists(FEATURES_PATH):
        # 先读取所有数据到内存
        with h5py.File(FEATURES_PATH, 'r') as hf:
            filenames = [f.decode('utf-8') for f in hf['filenames'][:]]
            features = hf['features'][:]
        
        # 确保文件已关闭后，再处理数据
        new_filenames = []
        new_features = []
        for i, filename in enumerate(filenames):
            if filename not in deleted_files:
                new_filenames.append(filename)
                new_features.append(features[i])
        
        # 现在可以安全地创建新文件
        if new_filenames:
            new_features = np.array(new_features)
            try:
                with h5py.File(FEATURES_PATH, 'w') as new_hf:
                    new_hf.create_dataset("features", data=new_features)
                    new_hf.create_dataset("filenames", 
                                        data=np.array(new_filenames, dtype=h5py.string_dtype(encoding='utf-8')))
            except Exception as e:
                logging.error(f"创建新特征文件失败: {str(e)}")
                raise HTTPException(500, "无法更新特征数据库")
        else:
            os.remove(FEATURES_PATH)

    save_trash([])
    return JSONResponse({
        "deleted": deleted_files,
        "remaining": len(os.listdir(image_dir)) if os.path.exists(image_dir) else 0
    })

@app.post("/api/search")
async def search_similar(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "仅支持图片文件")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        query_feat = extract_features(temp_path)
        
        # Check if feature database exists
        if not os.path.exists(FEATURES_PATH):
            return {"results": []}
            
        with h5py.File(FEATURES_PATH, 'r') as hf:
            all_features = hf['features'][:]
            filenames = hf['filenames'][:]
            
            # If no features available, return empty
            if len(all_features) == 0:
                return {"results": []}
                
            # Calculate similarities
            query_norm = np.linalg.norm(query_feat)
            features_norm = np.linalg.norm(all_features, axis=1)
            
            # Handle cases where norm is zero (avoid division by zero)
            valid_indices = features_norm > 0
            valid_features = all_features[valid_indices]
            valid_filenames = filenames[valid_indices]
            valid_norms = features_norm[valid_indices]
            
            if len(valid_features) == 0:
                return {"results": []}
                
            similarities = np.dot(valid_features, query_feat) / (valid_norms * query_norm)
            
            # Ensure all similarities are valid numbers
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
            
            top_indices = np.argsort(similarities)[-12:][::-1]
            results = [{
                "filename": valid_filenames[i].decode('utf-8'),
                "similarity": float(similarities[i])
            } for i in top_indices]
            
            return {"results": results}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/check_file/{filename}")
async def check_file_exists(filename: str):
    file_path = IMAGE_DIR + '/' + filename
    return {
        "exists": os.path.exists(file_path),
        "path": str(file_path),
        "readable": os.access(file_path, os.R_OK)
    }

@app.get("/", response_class=HTMLResponse)
async def main():
    return open("index.html").read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config=None  # 禁用Uvicorn默认日志配置
    )