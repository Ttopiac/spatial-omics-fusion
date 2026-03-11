"""
Download DLPFC (Dorsolateral Prefrontal Cortex) spatial transcriptomics dataset.

What is this data?
- Brain tissue slices where we measure gene activity at thousands of spatial locations ("spots")
- Each spot has: a gene expression vector (~33K genes) + an (x,y) coordinate on the tissue
- Each spot is labeled with a "spatial domain" (brain layer) — this is our prediction target
- 12 slices total, ~3500-4800 spots each

Think of it as: a 2D scatter plot of points, where each point has a 33K-dim feature vector
and a class label. Our job is to classify the points using both the features AND their
spatial arrangement.

Source: Figshare (CellCharter pre-processed), originally from Maynard et al., Nature Neuroscience 2021.
"""
import argparse
import os
import urllib.request

# All 12 DLPFC slice IDs
ALL_SAMPLE_IDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

# Direct download URLs from Figshare (pre-processed h5ad with labels + coordinates)
FIGSHARE_URLS = {
    "151507": "https://ndownloader.figshare.com/files/39055556",
    "151508": "https://ndownloader.figshare.com/files/39055589",
    "151509": "https://ndownloader.figshare.com/files/39055586",
    "151510": "https://ndownloader.figshare.com/files/39055583",
    "151669": "https://ndownloader.figshare.com/files/39055580",
    "151670": "https://ndownloader.figshare.com/files/39055577",
    "151671": "https://ndownloader.figshare.com/files/39055574",
    "151672": "https://ndownloader.figshare.com/files/39055571",
    "151673": "https://ndownloader.figshare.com/files/39055568",
    "151674": "https://ndownloader.figshare.com/files/39055565",
    "151675": "https://ndownloader.figshare.com/files/39055562",
    "151676": "https://ndownloader.figshare.com/files/39055559",
}


def download_single(sample_id: str, output_dir: str) -> str:
    """Download a single DLPFC slice from Figshare."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sample_id}.h5ad")

    if os.path.exists(output_path):
        print(f"  [{sample_id}] Already exists: {output_path}")
        return output_path

    url = FIGSHARE_URLS[sample_id]
    print(f"  [{sample_id}] Downloading from Figshare ...")
    urllib.request.urlretrieve(url, output_path)
    print(f"  [{sample_id}] Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download DLPFC dataset")
    parser.add_argument("--sample_id", type=str, default="151673",
                        help="Sample ID to download (default: 151673)")
    parser.add_argument("--all", action="store_true",
                        help="Download all 12 slices")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Output directory")
    args = parser.parse_args()

    if args.all:
        for sid in ALL_SAMPLE_IDS:
            download_single(sid, args.output_dir)
    else:
        download_single(args.sample_id, args.output_dir)


if __name__ == "__main__":
    main()
