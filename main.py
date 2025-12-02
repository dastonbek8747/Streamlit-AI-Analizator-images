import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests
from pathlib import Path
import time
import json


class AdvancedImageAnalyzer:
    """Rasmlarni batafsil tahlil qiluvchi sistema"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_model = None
        self.classification_model = None
        self.transform = None
        self.label_map = None
        self._initialize()

    def _initialize(self):
        """Modellarni yuklash"""
        # Klassifikatsiya modeli (ResNet-50)
        self.classification_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.classification_model.to(self.device)
        self.classification_model.eval()

        # Obyekt aniqlash modeli (Faster R-CNN)
        self.detection_model = models.detection.fasterrcnn_resnet50_fpn(
            weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.detection_model.to(self.device)
        self.detection_model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # COCO labellarini yuklash
        self.coco_labels = self._get_coco_labels()

        # ImageNet labellarini yuklash
        self.label_map = self._load_imagenet_labels()

    def _get_coco_labels(self):
        """COCO dataset kategoriyalari (o'zbek tiliga tarjima qilingan)"""
        return {
            1: 'odam', 2: 'velosiped', 3: 'mashina', 4: 'mototsikl', 5: 'samolyot',
            6: 'avtobus', 7: 'poyezd', 8: 'yuk mashinasi', 9: 'qayiq', 10: 'yo\'roqnoma',
            11: 'yong\'in gidrant', 13: 'to\'xtash belgisi', 14: 'parkometr', 15: 'skameyka',
            16: 'qush', 17: 'mushuk', 18: 'it', 19: 'ot', 20: 'qo\'y',
            21: 'sigir', 22: 'fil', 23: 'ayiq', 24: 'zebra', 25: 'jirafa',
            27: 'ryukzak', 28: 'soyabon', 31: 'sumka', 32: 'galstuk', 33: 'chemodan',
            34: 'frisbee', 35: 'chang\'i', 36: 'snoubord', 37: 'sport to\'pi', 38: 'uçurtma',
            39: 'beysbol tayoqchasi', 40: 'beysbol qo\'lqop', 41: 'skeytbord', 42: 'sorf taxtasi',
            43: 'tennis raketasi', 44: 'shisha', 46: 'piyola', 47: 'vilka', 48: 'pichoq',
            49: 'qoshiq', 50: 'kosa', 51: 'banan', 52: 'olma', 53: 'sendvich',
            54: 'apelsin', 55: 'brokkoli', 56: 'sabzi', 57: 'hot-dog', 58: 'pizza',
            59: 'donut', 60: 'tort', 61: 'stul', 62: 'divan', 63: 'o\'simlik',
            64: 'yotoq', 67: 'stol', 70: 'hojatxona', 72: 'televizor', 73: 'noutbuk',
            74: 'sichqoncha', 75: 'pult', 76: 'klaviatura', 77: 'telefon', 78: 'mikroto\'lqinli pech',
            79: 'duxovka', 80: 'toster', 81: 'lavabo', 82: 'muzlatgich', 84: 'kitob',
            85: 'soat', 86: 'guldon', 87: 'qaychi', 88: 'ayiqcha', 89: 'fen',
            90: 'tish cho\'tkasi'
        }

    def _load_imagenet_labels(self):
        """ImageNet labellarini yuklash"""
        try:
            url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            labels = response.json()
            return labels
        except:
            return [f"Kategoriya {i}" for i in range(1000)]

    def detect_objects(self, image, threshold=0.5):
        """Rasmdagi obyektlarni aniqlash"""
        try:
            # Rasmni tensorga aylantirish
            img_tensor = transforms.ToTensor()(image).to(self.device)

            # Prediksiya
            with torch.no_grad():
                predictions = self.detection_model([img_tensor])

            # Natijalarni qayta ishlash
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            # Threshold bo'yicha filtrlash
            detected_objects = []
            for box, label, score in zip(boxes, labels, scores):
                if score >= threshold:
                    obj_name = self.coco_labels.get(label, f"Obyekt {label}")
                    detected_objects.append({
                        'name': obj_name,
                        'confidence': float(score * 100),
                        'box': box.tolist()
                    })

            return detected_objects
        except Exception as e:
            st.error(f"Obyekt aniqlashda xato: {str(e)}")
            return []

    def analyze_colors(self, image):
        """Rasmning asosiy ranglarini aniqlash"""
        try:
            # Rasmni kichiklashtirish (tezroq ishlash uchun)
            img_small = image.resize((100, 100))
            pixels = list(img_small.getdata())

            # Ranglarni hisoblash
            from collections import Counter
            color_counts = Counter(pixels)
            most_common = color_counts.most_common(5)

            # Rang nomlarini aniqlash
            colors = []
            for color, count in most_common:
                color_name = self._get_color_name(color)
                percentage = (count / len(pixels)) * 100
                colors.append({
                    'name': color_name,
                    'rgb': color,
                    'percentage': percentage
                })

            return colors
        except Exception as e:
            st.error(f"Rang tahlilida xato: {str(e)}")
            return []

    def _get_color_name(self, rgb):
        """RGB qiymatidan rang nomini aniqlash"""
        r, g, b = rgb[:3] if len(rgb) >= 3 else (rgb[0], rgb[0], rgb[0])

        # Oddiy rang klassifikatsiyasi
        if r > 200 and g > 200 and b > 200:
            return "oq"
        elif r < 50 and g < 50 and b < 50:
            return "qora"
        elif r > 200 and g < 100 and b < 100:
            return "qizil"
        elif r < 100 and g > 200 and b < 100:
            return "yashil"
        elif r < 100 and g < 100 and b > 200:
            return "ko'k"
        elif r > 200 and g > 200 and b < 100:
            return "sariq"
        elif r > 200 and g < 100 and b > 200:
            return "pushti"
        elif r < 100 and g > 200 and b > 200:
            return "ko'kish-yashil"
        elif r > 150 and g > 100 and b < 100:
            return "jigarrang"
        elif r > 150 and g > 150 and b > 150:
            return "kulrang"
        else:
            return "aralash rang"

    def classify_image(self, image, top_k=5):
        """Umumiy klassifikatsiya"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.classification_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            top_probs, top_indices = torch.topk(probabilities, top_k)

            results = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.label_map[idx.item()] if idx.item() < len(self.label_map) else f"Kategoriya {idx.item()}"
                results.append((label, prob.item() * 100))

            return results
        except Exception as e:
            st.error(f"Klassifikatsiya xatosi: {str(e)}")
            return []

    def generate_description(self, image):
        """Rasmni batafsil tavsif qilish"""
        # Obyektlarni aniqlash
        objects = self.detect_objects(image)

        # Ranglarni tahlil qilish
        colors = self.analyze_colors(image)

        # Klassifikatsiya
        classification = self.classify_image(image, top_k=3)

        # Tavsif yaratish
        description = "Bu rasmda "

        if objects:
            obj_names = [obj['name'] for obj in objects[:5]]
            description += f"{len(objects)} ta obyekt aniqlandi: {', '.join(obj_names)}. "

        if classification:
            top_class = classification[0][0]
            description += f"Asosiy kategoriya: {top_class}. "

        if colors:
            top_colors = [c['name'] for c in colors[:3]]
            description += f"Asosiy ranglar: {', '.join(top_colors)}."

        return {
            'description': description,
            'objects': objects,
            'colors': colors,
            'classification': classification
        }


def main():
    st.set_page_config(
        page_title="Ilg'or Rasm Tahlil Tizimi",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Ilg'or Rasm Tahlil Tizimi")
    st.markdown("""
    Bu tizim rasmlarni **chuqur tahlil** qiladi:
    - 🎯 Obyektlarni aniqlash
    - 🎨 Ranglarni tahlil qilish
    - 📝 Batafsil tavsif yaratish
    - 👤 Odamlarni tanish
    """)

    # Modelni yuklash
    if 'analyzer' not in st.session_state:
        with st.spinner('🔄 Modellar yuklanmoqda...'):
            st.session_state.analyzer = AdvancedImageAnalyzer()
        st.success('✅ Modellar tayyor!')

    analyzer = st.session_state.analyzer

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Sozlamalar")
        detection_threshold = st.slider(
            "Aniqlash chegarasi",
            0.1, 1.0, 0.5, 0.05
        )
        show_boxes = st.checkbox("Obyekt chegaralarini ko'rsatish", value=True)

    # Rasm yuklash
    uploaded_file = st.file_uploader(
        "📁 Rasm yuklang",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp']
    )

    # Demo rasmlar
    st.markdown("### 🎨 Demo rasmlar:")
    col1, col2, col3 = st.columns(3)

    demo_urls = {
        "Odamlar": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=500",
        "Hayvonlar": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=500",
        "Manzara": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500"
    }

    selected_demo = None
    with col1:
        if st.button("👥 Odamlar"):
            selected_demo = demo_urls["Odamlar"]
    with col2:
        if st.button("🐾 Hayvonlar"):
            selected_demo = demo_urls["Hayvonlar"]
    with col3:
        if st.button("🏔️ Manzara"):
            selected_demo = demo_urls["Manzara"]

    # Rasmni qayta ishlash
    image_to_process = None

    if uploaded_file:
        image_to_process = Image.open(uploaded_file)
    elif selected_demo:
        try:
            response = requests.get(selected_demo, timeout=10)
            from io import BytesIO
            image_to_process = Image.open(BytesIO(response.content))
        except:
            st.error("Demo rasm yuklanmadi.")

    # Natijalarni ko'rsatish
    if image_to_process:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Asl Rasm")
            st.image(image_to_process, use_container_width=True)

        with col2:
            st.subheader("📊 Tahlil Natijalari")

            with st.spinner('🔍 Tahlil qilinmoqda...'):
                start_time = time.time()
                analysis = analyzer.generate_description(image_to_process)
                elapsed_time = time.time() - start_time

            st.success(f"✅ Tayyor! ({elapsed_time:.2f}s)")

            # Tavsif
            st.markdown("### 📝 Umumiy Tavsif")
            st.info(analysis['description'])

            # Aniqlangan obyektlar
            if analysis['objects']:
                st.markdown("### 🎯 Aniqlangan Obyektlar")
                for i, obj in enumerate(analysis['objects'][:10], 1):
                    st.markdown(f"**{i}. {obj['name']}** - {obj['confidence']:.1f}% ishonch")

            # Ranglar
            if analysis['colors']:
                st.markdown("### 🎨 Asosiy Ranglar")
                for color in analysis['colors']:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{color['name'].capitalize()}** - {color['percentage']:.1f}%")
                    with col_b:
                        rgb = color['rgb']
                        st.color_picker("", f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", disabled=True,
                                        label_visibility="collapsed")

            # Klassifikatsiya
            if analysis['classification']:
                st.markdown("### 🏷️ Kategoriyalar")
                for label, prob in analysis['classification']:
                    st.progress(prob / 100)
                    st.caption(f"{label}: {prob:.1f}%")


if __name__ == "__main__":
    main()