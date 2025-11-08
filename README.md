# ElysiaAPI  <img src="https://github.com/user-attachments/assets/bc6d687c-dd26-4bcd-bcbf-71a8a5681bc3" width="25"/>

Projeto de Visão Computacional usando Roboflow, Python e YOLOv8 para identificar motos em vídeos e detectar se estão em uso ou paradas, registrando eventos no banco para entender o fluxo real do pátio.

Além disso, foi integrado um protótipo IoT no Wokwi (ESP32) que monitora temperatura do motor.

Juntos, visão + IoT criam uma prova de conceito que mostra como é possível automatizar a gestão do pátio, entendendo não só quantas motos estão lá e o que estão fazendo, mas também o estado físico delas em tempo real.

---

## 👥 Integrantes

- **Iris Tavares Alves** - 557728 - 2TDSPM  
- **Taís Tavares Alves** - 557553 - 2TDSPM

---

## 🎬 Projeto Wokwi

> <a href="https://wokwi.com/projects/447006721048213505">Sensor Temperatura</a>

---

## ⚙️ Tecnologias Utilizadas

```text
- Python 3.12
- Ultralytics YOLOv8
- OpenCV
- Roboflow
- Oracledb
- ESP32
- Wokwi
```

---

### 1 - Clone o repositório
```text
git clone https://github.com/Irissuu/Elysia_iot.git
```

### 2 - Instale as dependências
```text
pip install ultralytics opencv-python oracledb
```

### 3. Configure a string de conexão (adicione seu user e senha do oracle sql)
```text
oracle_user = ""                  
oracle_pw   = ""               
oracle_dsn  = "oracle.fiap.com.br:1521/orcl"  
```

### 4 - Rode o projeto para detectar vagas
```text
python elysia_estacionamento.py
```

### 4.1 - Rode o projeto para detectar motos
```text
python elysia_motos.py
```

### 5 - Encerrar
```text
Pressione Q para encerrar a exibição do vídeo
```
---

## 📅 Resultados parciais - Roboflow + Wokwi
### ▸ Detecção de vagas
<img width="1904" height="1048" alt="image" src="https://github.com/user-attachments/assets/3f948121-c852-4dc7-9c5b-769bbd6b5b64" />

### ▸ Detecção de motos
<img width="1909" height="1042" alt="image" src="https://github.com/user-attachments/assets/bd96121b-4156-462b-90eb-7cfd5317feed" />

<img width="1277" height="711" alt="image" src="https://github.com/user-attachments/assets/318dcb84-870b-4760-ad9e-07060d66cfab" />

### ▸ Estado das motos
#### Parada
<img width="1276" height="716" alt="Captura de tela 2025-11-07 234318" src="https://github.com/user-attachments/assets/7a27b6df-1bfa-4eed-a671-a1f1f4c90fe8" />

#### Em uso
<img width="1278" height="720" alt="Captura de tela 2025-11-07 234357" src="https://github.com/user-attachments/assets/cef6ba59-4535-4820-b9ad-96b94a6c691c" />

### ▸ Temperatura do motor em °C
#### Normal
<img width="1020" height="484" alt="image" src="https://github.com/user-attachments/assets/fd67f976-d404-4efd-9ed2-6a34dd8571dd" />

#### Superaquecimento
<img width="1040" height="477" alt="image" src="https://github.com/user-attachments/assets/d5de5e95-4dbe-445f-be3b-992d45a68859" />





