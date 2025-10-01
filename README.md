# ElysiaAPI  <img src="https://github.com/user-attachments/assets/bc6d687c-dd26-4bcd-bcbf-71a8a5681bc3" width="25"/>

Projeto de Visão Computacional desenvolvido com Roboflow, Python e YOLOv8.

O sistema possui dois modelos de detecção:
- Vagas ocupadas e vagas livres → baseado em um vídeo simulado com visão aérea de estacionamento.
- Motos → identificação automática de motocicletas em vídeos, permitindo contabilizar e monitorar a ocupação do pátio.

Este protótipo serve como uma prova de conceito para aprimorar o controle e monitoramento de estacionamentos e pátios, ajudando na gestão inteligente das vagas e dos veículos.

---

## 👥 Integrantes

- **Iris Tavares Alves** - 557728 - 2TDSPM  
- **Taís Tavares Alves** - 557553 - 2TDSPM

---

## 🎬 Vídeo

> <a href="https://youtu.be/SA1OJPfUA78?si=qyKKehPVyFEFiIsz">Vídeo</a>
---

## ⚙️ Tecnologias Utilizadas

```text
- Python 3.12
- Ultralytics YOLOv8
- OpenCV
- Roboflow 
```

### 1 - Clone o repositório
```text
git clone https://github.com/Irissuu/Elysia_iot.git
```

### 2 - Instale as dependências
```text
pip install ultralytics opencv-python
```

### 3 - Rode o projeto para detectar vagas
```text
python elysia_estacionamento.py
```

### 3.1 - Rode o projeto para detectar motos
```text
python elysia_motos.py
```

### 4 - Encerrar
```text
Pressione Q para encerrar a exibição do vídeo
```

## 📅 Resultados parciais
### ▸ Detecção de vagas
<img width="1904" height="1048" alt="image" src="https://github.com/user-attachments/assets/3f948121-c852-4dc7-9c5b-769bbd6b5b64" />

### ▸ Detecção de Motos
<img width="1909" height="1042" alt="image" src="https://github.com/user-attachments/assets/bd96121b-4156-462b-90eb-7cfd5317feed" />

<img width="1277" height="711" alt="image" src="https://github.com/user-attachments/assets/318dcb84-870b-4760-ad9e-07060d66cfab" />






