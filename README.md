# Robust Pareto Design of GaN HEMTs for Millimeter-Wave Applications

This repository accompanies the manuscript [Robust Pareto Design of GaN HEMTs for Millimeter-Wave Applications](https://arxiv.org/abs/2406.17337).

**Abstract:**<br> 
This paper introduces a robust Pareto design approach for selecting Gallium Nitride (GaN) High Electron Mobility Transistors (HEMTs), particularly for power amplifier (PA) and low-noise amplifier (LNA) designs in 5G applications. We consider five key design variables and two settings (PAs and LNAs) where we have multiple objectives. We assess designs based on three critical objectives, evaluating each by its worst-case performance across a range of Gate-Source Voltages ($V_{\text{GS}}$). We conduct simulations across a range of ($V_{\text{GS}}$) values to ensure a thorough and robust analysis. For PAs, the optimization goals are to maximize the worst-case modulated average output power ($P_{\text{out,avg}}$) and power-added efficiency ($PAE_{\text{avg}}$) while minimizing the worst-case average junction temperature ($T_{\text{j,avg}}$) under a modulated 64-QAM signal stimulus. In contrast, for LNAs, the focus is on maximizing the worst-case maximum oscillation frequency ($f_{\text{max}}$) and Gain, and minimizing the worst-case minimum noise figure ($NF_{\text{min}}$). We utilize a derivative-free optimization method to effectively identify robust Pareto optimal device designs. This approach enhances our comprehension of the trade-off space, facilitating more informed decision-making. Furthermore, this method is general across different applications. Although it does not guarantee a globally optimal design, we demonstrate its effectiveness in GaN device sizing. The primary advantage of this method is that it enables the attainment of near-optimal or even optimal designs with just a fraction of the simulations required for an exhaustive full-grid search.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- [HOLA](https://github.com/blackrock/HOLA)

You can install the HOLA library using pip:

```sh
pip install git+https://github.com/blackrock/HOLA.git

