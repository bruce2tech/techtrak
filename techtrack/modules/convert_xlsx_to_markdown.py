import pandas as pd

all_sheets = pd.read_excel("model_comparison_metrics2.xlsx", sheet_name=None)

for sheet_name, sheet_df in all_sheets.items():
    md = sheet_df.to_markdown(index=False)
    out_file = f"{sheet_name.replace(' ', '_')}.md"  # make a safe filename
    with open(out_file, "w") as f:
        # write a title
        f.write(f"# {sheet_name}\n\n")
        # now write the actual markdown table
        f.write(md)
    print(f"→ Wrote {out_file}")
