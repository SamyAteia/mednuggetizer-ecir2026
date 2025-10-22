"""
Demonstration script showing the complete nugget extraction workflow.
This demo uses mock data to show how the system works.
"""

print("=" * 70)
print("MEDICAL INFORMATION NUGGET EXTRACTION TOOL - DEMONSTRATION")
print("=" * 70)
print()

# Step 1: Show AutoNuggetizer
print("STEP 1: AutoNuggetizer - Extracting nuggets from text")
print("-" * 70)

from auto_nuggetizer import AutoNuggetizer

nuggetizer = AutoNuggetizer(seed=42)
sample_text = """
Patient presents with symptoms of type 2 diabetes mellitus including 
increased thirst, frequent urination, and fatigue. Laboratory tests 
confirm elevated blood glucose levels at 180 mg/dL fasting. HbA1c 
measured at 7.8% indicates poor glycemic control over past 3 months.

Treatment plan initiated with metformin 500mg twice daily, dietary 
modifications emphasizing low glycemic index foods, and exercise 
program of 30 minutes daily walking. Patient education provided on 
blood glucose monitoring and hypoglycemia recognition.

Follow-up appointment scheduled in 4 weeks to assess treatment 
response and adjust medications as needed. Referral to endocrinologist 
recommended if glycemic control not achieved within 3 months.
"""

print("\nInput text (excerpt):")
print(sample_text[:200] + "...")

print("\nRunning extraction 3 times to test reproducibility...")
all_runs = nuggetizer.run_multiple_times(sample_text, n=3)

for i, nuggets in enumerate(all_runs, 1):
    print(f"\nRun {i}: Extracted {len(nuggets)} nuggets")
    for j, nugget in enumerate(nuggets[:3], 1):
        print(f"  {j}. {nugget[:60]}...")

# Step 2: Show what the complete pipeline does
print("\n" + "=" * 70)
print("STEP 2: Complete Pipeline Overview")
print("-" * 70)

print("""
The complete system performs these steps:

1. UPLOAD: Accept multiple PDF files
2. EXTRACT: Run AutoNuggetizer n times per PDF (e.g., n=3)
3. EMBED: Convert nuggets to numerical vectors (TF-IDF)
4. CLUSTER: Group similar nuggets together
5. FILTER: Keep only clusters with exactly n nuggets (one from each run)
   → This ensures reproducibility!
6. MERGE: Combine filtered nuggets across all PDFs
7. RECLUSTER: Find patterns across documents
8. RANK: Sort clusters by size (larger = stronger evidence)
9. DISPLAY: Show ranked results in web UI or API

Key Innovation: Only keeping clusters with exactly n nuggets (one per run)
ensures we only report information that appears consistently across runs,
supporting reproducibility in LLM-based extraction.
""")

# Step 3: Show clustering concept
print("=" * 70)
print("STEP 3: Clustering and Filtering Concept")
print("-" * 70)

print("""
Example: Running extraction 3 times on one PDF:

Run 1: ["diabetes diagnosis", "metformin treatment", "blood glucose 180"]
Run 2: ["type 2 diabetes confirmed", "metformin prescribed", "glucose elevated"]
Run 3: ["diabetes mellitus type 2", "metformin therapy", "high blood sugar"]

After clustering:
  Cluster A (3 nuggets): diabetes-related ✓ (one from each run - KEEP)
  Cluster B (3 nuggets): metformin-related ✓ (one from each run - KEEP)
  Cluster C (3 nuggets): glucose-related ✓ (one from each run - KEEP)
  Cluster D (1 nugget): follow-up ✗ (not in all runs - DISCARD)

Result: Only reproducible nuggets are retained!
""")

# Step 4: Show evidence ranking
print("=" * 70)
print("STEP 4: Evidence Strength Ranking")
print("-" * 70)

print("""
After merging across PDFs and reclustering:

Cluster 1 (12 nuggets from 4 PDFs): "Metformin as first-line treatment"
  → HIGH evidence strength (appears in multiple documents)
  
Cluster 2 (9 nuggets from 3 PDFs): "HbA1c monitoring importance"
  → MEDIUM-HIGH evidence strength
  
Cluster 3 (6 nuggets from 2 PDFs): "Lifestyle modifications"
  → MEDIUM evidence strength
  
Cluster 4 (3 nuggets from 1 PDF): "Insulin therapy"
  → LOW evidence strength

Ranking by cluster size provides evidence strength scoring!
""")

# Step 5: API Usage Example
print("=" * 70)
print("STEP 5: Using the System")
print("-" * 70)

print("""
WEB UI:
1. Start server: python app.py
2. Open browser: http://localhost:5000
3. Upload PDFs, optionally add query
4. View ranked results with evidence strength

REST API:
curl -X POST http://localhost:5000/api/extract \\
  -F "files=@paper1.pdf" \\
  -F "files=@paper2.pdf" \\
  -F "query=diabetes treatment" \\
  -F "n_runs=3"

Returns JSON with:
{
  "status": "success",
  "n_pdfs": 2,
  "n_runs": 3,
  "total_nuggets": 24,
  "n_clusters": 5,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 8,
      "nuggets": ["...", "..."],
      "sources": ["paper1.pdf", "paper2.pdf"]
    }
  ]
}
""")

print("=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
print()
print("Next steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run tests: python test_system.py")
print("3. Start application: python app.py")
print("4. Open browser: http://localhost:5000")
print()
print("For more information, see README.md and SETUP.md")
