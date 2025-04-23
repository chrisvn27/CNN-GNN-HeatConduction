# Deep Learning for Heat Conduction Prediction

This project leverages **Convolutional Neural Networks (CNNs)** and **Graph Neural Networks (GNNs)** to predict steady-state temperature distributions from 2D finite difference simulations of heat conduction.

The models are trained on datasets where **thermal conductivity varies spatially**, allowing evaluation of how well each architecture generalizes to different conductivity patterns and boundary conditions.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data with varying conductivity:
   ```bash
   python src/Data_generation.py
   ```

3. Train models:
   - CNN:
     ```bash
     python src/CNN_training.py
     ```
   - GNN:
     ```bash
     python src/GNN_training.py
     ```

4. View results in the `figures/` folder.

---

## Notes
- **Datasets** (`.npy`, `.pt`) and **model checkpoints** (`.pth`) are excluded from this repo.
- Thermal conductivity variations follow predefined patterns (e.g., linear gradients).
- For datasets or pretrained models, contact me.

---

## Author
Christian Valencia Narva  
Master's in Mechanical Engineering — Deep Learning & Computational Simulations  
[LinkedIn](https://www.linkedin.com/in/christian-valencia3)
