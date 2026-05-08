import os
import torch
import numpy as np
import csv
from Bio.Data import IUPACData
from allosteric_analyzer import AllosticHeadAnalyzer
from scipy.stats import ttest_1samp
import random

def main():
    # Setting seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Adenosine A2A receptor 2YDO
    sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSAAAADILVGVLAIPFAIAISTGFCAACHGCLFIACFVLVLTASSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKAHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRSHVLRQQEPFKAAAAENLYFQ"
    allosteric_res = [169, 253, 277, 278]
    orthosteric_res = [] # se puede rellenar para representar residuos del sitio ortosterico
    pathway_res = []

    allosteric_res_3l = []
    one_to_three = IUPACData.protein_letters_1to3
    for i in allosteric_res:
        allosteric_res_3l.append(one_to_three[f"{sequence[i-1]}"])

    print("=" * 50)
    print(f"The allosteric residues are: ")
    for idx, site in enumerate(allosteric_res):
        print(allosteric_res_3l[idx], site)

    # Initialize analyzer
    analyzer = AllosticHeadAnalyzer(threshold=0.3)

    # Basic analysis
    results = analyzer.analyze_protein(sequence, allosteric_res)

    impact_scores_tensor = results["impacts"]
    snr_values_tensor = results["snrs"]
    p_values_tensor = results["p_values"]

    assert impact_scores_tensor.ndim == 2
    assert snr_values_tensor.ndim == 2

    num_layers, num_heads = impact_scores_tensor.shape

    print(f"\nNumber of layers: {num_layers}")
    print(f"Number of heads: {num_heads}")

    head_stats = []
    for layer in range(num_layers):
        for head in range(num_heads):
            impact = impact_scores_tensor[layer][head].item()
            snr = snr_values_tensor[layer][head].item()
            head_stats.append((layer, head, impact, snr))

    print("\nAllosteric sensitivity analysis per attention head:")
    print("Layer | Head | Impact Score | SNR")
    print("-" * 40)
    for layer, head, impact, snr in head_stats:
        print(f"{layer:5d} | {head:4d} | {impact:11.3f} | {snr:6.2f}")

    # Apply t-test for significance based on null hypothesis
    impacts = np.array([stat[2] for stat in head_stats])
    t_stats = []
    for layer, head, impact, snr in head_stats:
        t_stat, p_val = ttest_1samp(impacts, impact, alternative='less')
        t_stats.append((layer, head, impact, snr, p_val))

    mean_impact = np.mean([stat[2] for stat in head_stats])
    mean_snr = np.mean([stat[3] for stat in head_stats])

    sensitive_heads = [
    (layer, head) for (layer, head, impact, snr, p_val) in t_stats
    if p_val < 0.01 and snr > 2.0
    ]

    print(f"\nMost sensitive heads to allosteric sites {allosteric_res}:")
    print(f"(p < 0.01 and SNR > 2.0)")
    print(f"(Layer, Head) pairs: {sensitive_heads}")
    # Get attention maps
    attention_maps = analyzer.get_attention_maps(sequence)  # [1, layers, heads, seq_len, seq_len]

    output_dir = "attention_data"
    os.makedirs(output_dir, exist_ok=True)

    # Cálculo de la atención promedio acumulada hacia residuos alostéricos
    accumulated_attention = analyzer.compute_accumulated_average_attention(
        attention_maps=attention_maps,
        sequence=sequence,
        allosteric_sites=allosteric_res,
        sensitive_heads=sensitive_heads
    )

    # Calcular el percentil 85
    attention_values = list(accumulated_attention.values())
    percentile_85 = np.percentile(attention_values, 95)

    csv_file = os.path.join(output_dir, "accumulated_attention_a2a.csv")
    with open(csv_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Residue", "Position", "AvgAccAtt"])

        for i in sorted(accumulated_attention.keys()):
            avg_attention = accumulated_attention[i]
            if avg_attention > percentile_85:
                residue = sequence[i - 1]
                writer.writerow([residue, i, avg_attention])

    print(f"Saved accumulated average attention values to {csv_file}")

    # Visualize each selected head
 # === NEW: Visualize mean of attention maps across sensitive heads ===
    if sensitive_heads:
        attention_stack = torch.stack([attention_maps[0, l, h] for (l, h) in sensitive_heads])  # [N, seq_len, seq_len]
        mean_attention_map = torch.mean(attention_stack, dim=0)  # [seq_len, seq_len]

        # Preparamos listas de impacto, SNR y p-valores correspondientes a las cabezas sensibles
        sensitive_impact_scores = [impact_scores_tensor[l][h] for (l, h) in sensitive_heads]
        sensitive_snr_values = [snr_values_tensor[l][h] for (l, h) in sensitive_heads]
        sensitive_p_values = [p_values_tensor[l][h] for (l, h) in sensitive_heads]  # No usados en visualización, pero pueden pasarse si se desea filtrar por p más adelante

        # analyzer.visualize_average_head_attention(
        #     attention_maps=attention_maps,
        #     impact_scores=sensitive_impact_scores,
        #     snr_values=sensitive_snr_values,
        #     allosteric_sites=allosteric_res,
        #     orthosteric_sites=orthosteric_res,
        #     pathway_sites=pathway_res,
        #     sequence=sequence,
        #     p_values=sensitive_p_values,
        #     cmap = "viridis" # default: "viridis"
        # )
    # Save sensitive head data
    head_info = {
        'sensitive_heads': sensitive_heads,
        'impact_scores': {f"{l}_{h}": impact_scores_tensor[l][h].item() for (l, h) in sensitive_heads},
        'snr_values': {f"{l}_{h}": snr_values_tensor[l][h].item() for (l, h) in sensitive_heads}
    }
    torch.save(head_info, os.path.join(output_dir, 'sensitive_heads.pt'))

    if sensitive_heads:
        sensitive_attention = [attention_maps[0, l, h].unsqueeze(0) for (l, h) in sensitive_heads]
        sensitive_attention_tensor = torch.cat(sensitive_attention, dim=0)  # [N, seq_len, seq_len]
        torch.save(sensitive_attention_tensor, os.path.join(output_dir, 'sensitive_attention_maps.pt'))

    print(f"Saved all output data to directory: {output_dir}/")

if __name__ == "__main__":
    main()