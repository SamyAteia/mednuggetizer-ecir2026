"""
Flask web application for medical information nugget extraction.
Provides both a web UI and REST API.
"""
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
import os
import json
import uuid
import pandas as pd
import logging
import traceback
import multiprocessing
from threading import Lock
from backend.auto_nuggetizer_ginger import *
from backend.clustering_ginger import *
from backend.summarizers import *
from backend.utils import load_api_key
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# EXPERIMENT: Using threads instead of processes for clustering
# Threads share memory (one embedding model copy) and are simpler
# If this causes OpenMP issues, switch back to spawn or try forkserver
# 
# To revert to spawn mode, uncomment:
# multiprocessing.set_start_method('spawn', force=True)


def cluster_and_summarize(file_name, information_nuggets, query, n_runs, llm_confidence, api_key):
    """Cluster and summarize nuggets for a single PDF.
    
    Args:
        file_name: Name of the PDF file
        information_nuggets: List of extracted nuggets
        query: User query
        n_runs: Number of extraction runs
        llm_confidence: Confidence threshold
        api_key: API key string (picklable, unlike the client object)
        
    Returns:
        Tuple of (file_name, list of (summary, confidence) tuples)
    """
    min_cluster_size = max(1, int(n_runs * llm_confidence))

    if len(information_nuggets) >= min_cluster_size:
        bertopic = BERTopicClustering(min_cluster_size=min_cluster_size)
        bertopic_freq = bertopic.cluster(information_nuggets)
        clustered_docs = bertopic_freq.groupby('Topic')['Document'].apply(list).to_dict()
        clustered_docs.pop(-1, None)

        # Create summarizer in worker process (can't pickle genai.Client)
        nugget_summarizer = GPTSummarizerAllNuggets(api_key=api_key)

        results = []
        with ThreadPoolExecutor(max_workers=4) as summarizer_pool:
            futures = []
            for cluster, nuggets in clustered_docs.items():
                futures.append(summarizer_pool.submit(
                    nugget_summarizer.summarize_info_nuggets,
                    info_nuggets=nuggets,
                    query=query
                ))
            for f in as_completed(futures):
                summary = f.result()
                results.append((summary, min(len(nuggets)/n_runs, 1)))
        return file_name, results
    else:
        return file_name, []


LOGGING = True
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'

# Enable Flask request logging
app.logger.setLevel(logging.INFO)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for shared resources (initialized in main process only)
API_KEY = None
SHARED_NUGGET_DETECTOR = None
SHARED_NUGGET_SUMMARIZER = None
SHARED_HEADING_SUMMARIZER = None


def initialize_resources():
    """Initialize resources needed by the Flask process.
    
    With threading: All threads share these resources (memory efficient).
    With multiprocessing: Only MainProcess uses these (workers create their own).
    """
    global API_KEY, SHARED_NUGGET_DETECTOR, SHARED_NUGGET_SUMMARIZER, SHARED_HEADING_SUMMARIZER
    
    # Skip initialization if already done
    if API_KEY is not None:
        return
    
    logger.info("=" * 60)
    logger.info("Initializing application resources...")
    logger.info("=" * 60)

    # 1. Load API key
    logger.info("Loading API key...")
    API_KEY = load_api_key()
    logger.info("✓ API key loaded")

    # 2. Initialize Gemini clients
    logger.info("Initializing Gemini clients...")
    SHARED_NUGGET_DETECTOR = GPTNuggetDetector(api_key=API_KEY)
    SHARED_NUGGET_SUMMARIZER = GPTSummarizerAllNuggets(api_key=API_KEY)
    SHARED_HEADING_SUMMARIZER = SummarizerHeadings(api_key=API_KEY)
    logger.info("✓ Gemini clients ready")

    # 3. Pre-load embedding model (threads will share this)
    logger.info("Initializing shared embedding model...")
    from backend.clustering_ginger import get_shared_embedding_model
    get_shared_embedding_model()
    logger.info("✓ Embedding model ready (shared across threads)")

    logger.info("=" * 60)
    logger.info("Application startup complete!")
    logger.info("=" * 60)


# Progress tracking
progress_store = {}
progress_lock = Lock()


# Initialize resources at application startup
# With threads: All workers share these resources
# With processes: Only MainProcess would use them
initialize_resources()


def send_progress(session_id, step, status, detail, percentage):
    """Send progress update for a specific session."""
    with progress_lock:
        if session_id not in progress_store:
            progress_store[session_id] = []
        progress_store[session_id].append({
            'step': step,
            'status': status,
            'detail': detail,
            'percentage': percentage
        })


@app.route('/')
def index():
    """Render the main web UI."""
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.svg', mimetype='image/svg+xml')


@app.route('/api/extract', methods=['POST'])
def extract_nuggets():
    """
    API endpoint to extract and cluster nuggets from uploaded PDFs.
    
    Expected form data:
    - files: Multiple PDF files
    - query (optional): Query string to guide extraction
    - n_runs (optional): Number of extraction runs (default: 3)
    - session_id (optional): Session ID for progress tracking
    
    Returns:
        JSON response with ranked clusters
    """
    session_id = None
    try:
        # Get session ID for progress tracking
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        logger.info(f"[{session_id}] Starting extraction request")
        
        # Initialize progress
        send_progress(session_id, 1, 'active', 'Validating files...', 2)
        
        # Get uploaded files
        if 'files' not in request.files:
            logger.error(f"[{session_id}] No files provided in request")
            return jsonify({
                'status': 'error',
                'message': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            logger.error(f"[{session_id}] No files selected")
            return jsonify({
                'status': 'error',
                'message': 'No files selected'
            }), 400
        
        # Validate file types
        for file in files:
            if not file.filename.endswith('.pdf'):
                logger.error(f"[{session_id}] Invalid file type: {file.filename}")
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid file type: {file.filename}. Only PDF files are supported.'
                }), 400
        
        logger.info(f"[{session_id}] Validated {len(files)} PDF files")
        send_progress(session_id, 1, 'completed', f'{len(files)} files validated', 5)
        
        # Get optional parameters
        query = request.form.get('query')
        n_runs = int(request.form.get('n_runs', 3))
        llm_confidence = float(request.form.get('llm_confidence', 0.8))
        
        logger.info(f"[{session_id}] Parameters: n_runs={n_runs}, llm_confidence={llm_confidence}, query={query}")
        
        # Validate n_runs
        if n_runs < 1 or n_runs > 10:
            logger.error(f"[{session_id}] Invalid n_runs: {n_runs}")
            return jsonify({
                'status': 'error',
                'message': 'n_runs must be between 2 and 10'
            }), 400

        # Validate llm_confidence
        if llm_confidence < 0 or llm_confidence > 1:
            logger.error(f"[{session_id}] Invalid llm_confidence: {llm_confidence}")
            return jsonify({
                'status': 'error',
                'message': 'llm_confidence must be between 0 and 1'
            }), 400

        min_cluster_size = max(1, int(n_runs * llm_confidence))

        if LOGGING:
            logger.info(f"[{session_id}] llm_confidence={llm_confidence}, min_cluster_size={min_cluster_size}")

        # Step 2: Extract nuggets
        send_progress(session_id, 2, 'active', 'Starting nugget extraction...', 7)
        
        pdf_files = {}
        total_extractions = len(files) * n_runs
        completed_extractions = 0

        def process_file(file):
            nonlocal completed_extractions
            try:
                pdf_file_content = file.read()
                logger.info(f"[{session_id}] Processing {file.filename} ({len(pdf_file_content)} bytes)")
                
                with ThreadPoolExecutor() as inner_executor:
                    # Use the shared nugget detector instead of creating a new one
                    futures = [inner_executor.submit(SHARED_NUGGET_DETECTOR.detect_nuggets, query, pdf_file_content)
                               for _ in range(n_runs)]
                    results = []
                    for f in as_completed(futures):
                        try:
                            result = f.result()
                            results.extend(result)
                            completed_extractions += 1
                            progress = 7 + int((completed_extractions / total_extractions) * 53)
                            send_progress(session_id, 2, 'active', 
                                        f'Extraction {completed_extractions}/{total_extractions} completed', 
                                        progress)
                        except Exception as e:
                            logger.error(f"[{session_id}] Error in extraction future for {file.filename}: {str(e)}", exc_info=True)
                            completed_extractions += 1
                
                logger.info(f"[{session_id}] Extracted {len(results)} nuggets from {file.filename}")
                return file.filename, results
            except Exception as e:
                logger.error(f"[{session_id}] Error processing file {file.filename}: {str(e)}", exc_info=True)
                return file.filename, []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_file, file) for file in files]
            for future in as_completed(futures):
                try:
                    filename, results = future.result()
                    pdf_files[filename] = results
                except Exception as e:
                    logger.error(f"[{session_id}] Error in process_file future: {str(e)}", exc_info=True)

        send_progress(session_id, 2, 'completed', f'{total_extractions} extractions completed', 60)

        if LOGGING:
            logger.info(f"[{session_id}] Total nuggets per PDF: {[(k, len(v)) for k, v in pdf_files.items()]}")

        # Check if any nuggets were extracted
        total_nuggets_extracted = sum(len(nuggets) for nuggets in pdf_files.values())
        logger.info(f"[{session_id}] Total nuggets extracted: {total_nuggets_extracted}")
        
        if total_nuggets_extracted == 0:
            logger.warning(f"[{session_id}] No nuggets extracted from any PDF")
            send_progress(session_id, 3, 'completed', 'No nuggets extracted', 75)
            send_progress(session_id, 4, 'completed', 'Skipped (no nuggets)', 85)
            send_progress(session_id, 5, 'completed', 'No summaries to generate', 100)
            
            return jsonify({
                'status': 'warning',
                'message': 'No information nuggets were extracted from the provided PDFs. This could mean:\n'
                          '1. The PDFs do not contain relevant information for the query\n'
                          '2. The query might be too specific or unclear\n'
                          '3. The PDF content could not be properly extracted\n\n'
                          'Try rephrasing your query or using different PDF files.',
                'n_runs': n_runs,
                'n_pdfs': len(pdf_files),
                'total_nuggets': 0,
                'n_clusters': 0,
                'clusters': [],
                'pdf_names': list(pdf_files.keys()),
                'session_id': session_id
            })

        # Step 3: Per-PDF clustering
        send_progress(session_id, 3, 'active', 'Clustering nuggets per PDF...', 62)
        
        important_nuggets = {}
        completed_pdfs = 0
        
        if min_cluster_size > 1:
            # Using ThreadPoolExecutor for memory efficiency (shared embedding model)
            # Workers share the embedding model loaded in initialize_resources()
            # If this causes OpenMP issues, switch to ProcessPoolExecutor with spawn mode
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(cluster_and_summarize, fn, nuggets, query, n_runs, llm_confidence, API_KEY)
                    for fn, nuggets in pdf_files.items()
                ]
                for f in as_completed(futures):
                    try:
                        fn, results = f.result()
                        important_nuggets[fn] = results
                        completed_pdfs += 1
                        progress = 62 + int((completed_pdfs / len(files)) * 13)
                        logger.info(f"[{session_id}] Clustered {fn}: {len(results)} clusters")
                        send_progress(session_id, 3, 'active', 
                                    f'Clustered {completed_pdfs}/{len(files)} PDFs', 
                                    progress)
                    except Exception as e:
                        logger.error(f"[{session_id}] Error in clustering future: {str(e)}", exc_info=True)
                        completed_pdfs += 1
        else:
            for file_name, information_nuggets in pdf_files.items():
                important_nuggets[file_name] = [(nugget, 1) for nugget in information_nuggets]
                completed_pdfs += 1
                progress = 62 + int((completed_pdfs / len(files)) * 13)
                send_progress(session_id, 3, 'active', 
                            f'Processed {completed_pdfs}/{len(files)} PDFs', 
                            progress)

        send_progress(session_id, 3, 'completed', f'Clustered {len(files)} PDFs', 75)

        if LOGGING:
            logger.info(f"[{session_id}] Important nuggets per PDF: {[(k, len(v)) for k, v in important_nuggets.items()]}")

        # Step 4: Global clustering
        send_progress(session_id, 4, 'active', 'Preparing global clustering...', 77)
        
        all_documents = []
        all_nuggets = []
        all_confidences = []
        
        try:
            for k, v in important_nuggets.items():
                if v:  # Check if list is not empty
                    all_documents.extend([k] * len(v))
                    all_nuggets.extend([n[0] for n in v])
                    all_confidences.extend([n[1] for n in v])
        except (IndexError, TypeError, ValueError) as e:
            logger.error(f"[{session_id}] Error building global data structures: {str(e)}", exc_info=True)
            logger.error(f"[{session_id}] important_nuggets structure: {important_nuggets}")
            send_progress(session_id, 4, 'error', f'Data structure error: {str(e)}', 0)
            return jsonify({
                'status': 'error',
                'message': f'Error processing nugget data: {str(e)}'
            }), 500

        if LOGGING:
            logger.info(f"[{session_id}] Global: {len(all_nuggets)} nuggets, {len(all_documents)} docs, {len(all_confidences)} confidences")

        # Check if we have any nuggets after per-PDF clustering
        if len(all_nuggets) == 0:
            logger.warning(f"[{session_id}] No nuggets after per-PDF clustering")
            send_progress(session_id, 4, 'completed', 'No nuggets to cluster', 85)
            send_progress(session_id, 5, 'completed', 'No summaries to generate', 100)
            
            return jsonify({
                'status': 'warning',
                'message': 'No information nuggets passed the clustering threshold. '
                          'Try lowering the LLM confidence parameter or increasing the number of runs.',
                'n_runs': n_runs,
                'n_pdfs': len(pdf_files),
                'total_nuggets': 0,
                'n_clusters': 0,
                'clusters': [],
                'pdf_names': list(pdf_files.keys()),
                'session_id': session_id
            })

        send_progress(session_id, 4, 'active', 'Running BERTopic clustering...', 80)

        if len(all_nuggets) > 1:
            bertopic = BERTopicClustering(min_cluster_size=2, n_components=1)
            bertopic_freq = bertopic.cluster(all_nuggets)
            
            # Validate data before assignment
            if len(bertopic_freq) != len(all_documents):
                logger.error(f"[{session_id}] Length mismatch: bertopic_freq={len(bertopic_freq)}, all_documents={len(all_documents)}")
                raise ValueError(f"Data length mismatch: {len(bertopic_freq)} vs {len(all_documents)}")
            
            bertopic_freq['Source'] = all_documents
            bertopic_freq['Confidence'] = all_confidences

            if LOGGING:
                logger.info(f"[{session_id}] BERTopic clustering complete: {len(bertopic_freq)} entries")

            clustered_data = (
                bertopic_freq
                .groupby('Topic')[['Document', 'Source', 'Confidence']]
                .agg(list)
                .to_dict(orient='index')
            )
            not_clustered = clustered_data.pop(-1, None)
            logger.info(f"[{session_id}] Grouped into {len(clustered_data)} clusters, {len(not_clustered['Document']) if not_clustered else 0} unclustered")
        else:
            clustered_data = {}
            not_clustered = {
                'Document': all_nuggets,
                'Source': all_documents,
                'Confidence': all_confidences
            }
            logger.info(f"[{session_id}] Single nugget, treating as unclustered")

        send_progress(session_id, 4, 'completed', 'Global clustering complete', 85)

        # Step 5: Generate summaries
        send_progress(session_id, 5, 'active', 'Generating cluster summaries...', 87)
        
        results = {
            'status': 'success',
            'n_runs': n_runs,
            'n_pdfs': len(pdf_files),
            'total_nuggets': 0,
            'n_clusters': 0,
            'clusters': [],
            'pdf_names': list(pdf_files.keys()),
            'session_id': session_id
        }

        def summarize_cluster(cluster_id, cluster_information):
            try:
                # Use the shared heading summarizer instead of creating a new one
                cluster_heading = SHARED_HEADING_SUMMARIZER.summarize_info_nuggets(
                    info_nuggets=cluster_information['Document']
                )
                return {
                    'cluster_id': cluster_id,
                    'cluster_heading': cluster_heading,
                    'nuggets': [
                        f"{cluster_information['Document'][i]} (Confidence: {cluster_information['Confidence'][i] * 100}%)"
                        for i in range(len(cluster_information['Document']))
                    ],
                    'sources': cluster_information['Source'],
                    'size': len(cluster_information['Document'])
                }
            except Exception as e:
                logger.error(f"[{session_id}] Error summarizing cluster {cluster_id}: {str(e)}", exc_info=True)
                return {
                    'cluster_id': cluster_id,
                    'cluster_heading': 'Error generating summary',
                    'nuggets': [],
                    'sources': [],
                    'size': 0
                }

        def summarize_single_doc(cluster_id, document, source, confidence):
            try:
                # Use the shared heading summarizer instead of creating a new one
                cluster_heading = SHARED_HEADING_SUMMARIZER.summarize_info_nuggets(info_nuggets=[document])
                return {
                    'cluster_id': cluster_id,
                    'cluster_heading': cluster_heading,
                    'nuggets': [f"{document} (Confidence: {confidence * 100}%)"],
                    'sources': [source],
                    'size': 1
                }
            except Exception as e:
                logger.error(f"[{session_id}] Error summarizing single doc {cluster_id}: {str(e)}", exc_info=True)
                return {
                    'cluster_id': cluster_id,
                    'cluster_heading': 'Error generating summary',
                    'nuggets': [],
                    'sources': [],
                    'size': 0
                }

        futures = []
        total_clusters = len(clustered_data) + (len(not_clustered['Document']) if not_clustered else 0)
        completed_summaries = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Parallelize clustered data
            for cluster_id, cluster_information in clustered_data.items():
                results['n_clusters'] += 1
                results['total_nuggets'] += len(cluster_information['Document'])
                futures.append(executor.submit(summarize_cluster, results['n_clusters'], cluster_information))

            # Parallelize unclustered documents
            if not_clustered is not None:
                documents = not_clustered['Document']
                sources = not_clustered['Source']
                confidences = not_clustered['Confidence']
                for i in range(len(documents)):
                    results['n_clusters'] += 1
                    results['total_nuggets'] += 1
                    futures.append(executor.submit(
                        summarize_single_doc,
                        results['n_clusters'],
                        documents[i],
                        sources[i],
                        confidences[i]
                    ))

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    cluster_result = future.result()
                    results['clusters'].append(cluster_result)
                    completed_summaries += 1
                    progress = 87 + int((completed_summaries / total_clusters) * 8)
                    send_progress(session_id, 5, 'active', 
                                f'Generated {completed_summaries}/{total_clusters} summaries', 
                                progress)
                except Exception as e:
                    logger.error(f"[{session_id}] Error in summary future: {str(e)}", exc_info=True)
                    completed_summaries += 1

        send_progress(session_id, 5, 'completed', 'All summaries generated', 100)

        logger.info(f"[{session_id}] Request complete: {results['n_clusters']} clusters, {results['total_nuggets']} nuggets")
        
        return jsonify(results)
        
    except Exception as e:
        error_msg = f'Error processing request: {str(e)}'
        error_trace = traceback.format_exc()
        logger.error(f"[{session_id or 'unknown'}] {error_msg}\n{error_trace}")
        
        if session_id:
            send_progress(session_id, 0, 'error', f'Error: {str(e)}', 0)
        
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'trace': error_trace if app.debug else None
        }), 500


@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    """Stream progress updates for a specific session using Server-Sent Events."""
    def generate():
        last_sent = 0
        timeout_counter = 0
        max_timeout = 300  # 5 minutes max wait
        
        while timeout_counter < max_timeout:
            with progress_lock:
                if session_id in progress_store:
                    updates = progress_store[session_id][last_sent:]
                    if updates:
                        for update in updates:
                            yield f"data: {json.dumps(update)}\n\n"
                            last_sent += 1
                            
                        # Check if we're done (step 5 completed or error)
                        last_update = updates[-1]
                        if (last_update['step'] == 5 and last_update['status'] == 'completed') or \
                           last_update['status'] == 'error':
                            # Clean up progress store after completion
                            del progress_store[session_id]
                            break
                        
                        timeout_counter = 0  # Reset timeout on activity
                    else:
                        timeout_counter += 1
                else:
                    timeout_counter += 1
            
            import time
            time.sleep(1)  # Poll every second
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'X-Accel-Buffering': 'no'
                   })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Medical Nugget Extraction API'
    })


@app.route('/api/info', methods=['GET'])
def api_info():
    """Return API information and usage instructions."""
    return jsonify({
        'service': 'Medical Nugget Extraction API',
        'version': '1.0.0',
        'description': 'Reproducibility-supporting tool for LLM-based information nugget extraction',
        'endpoints': {
            '/': 'Web UI',
            '/api/extract': {
                'method': 'POST',
                'description': 'Extract and cluster nuggets from PDFs',
                'parameters': {
                    'files': 'Multiple PDF files (required)',
                    'query': 'Optional query to guide extraction',
                    'n_runs': 'Number of extraction runs (2-10, default: 3)'
                }
            },
            '/api/health': {
                'method': 'GET',
                'description': 'Health check endpoint'
            },
            '/api/info': {
                'method': 'GET',
                'description': 'API information'
            }
        }
    })


if __name__ == '__main__':
    # Resources are already initialized at module load time
    app.run(debug=True, host='0.0.0.0', port=4000)
