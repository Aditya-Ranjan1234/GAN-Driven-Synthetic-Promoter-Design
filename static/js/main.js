/**
 * DNA Sequence GAN - Main JavaScript
 * 
 * This file contains the client-side logic for the DNA Sequence GAN web interface.
 */

$(document).ready(function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // Show alert message
    function showAlert(message, type = 'success') {
        const alertHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        $('#alert-container').html(alertHTML);
        
        // Auto-dismiss after 5 seconds
        setTimeout(function() {
            $('.alert').alert('close');
        }, 5000);
    }
    
    // Show loading spinner
    function showLoading(container) {
        const spinnerHTML = `
            <div class="spinner-container">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
        
        $(container).html(spinnerHTML);
    }
    
    // Update status indicators
    function updateStatus(statusType, value) {
        const statusElement = $(`#${statusType}-status`);
        statusElement.text(value);
        
        // Add appropriate styling based on status
        statusElement.removeClass('text-success text-danger text-warning');
        
        if (value.includes('Not')) {
            statusElement.addClass('text-danger');
        } else if (value.includes('In progress')) {
            statusElement.addClass('text-warning');
        } else {
            statusElement.addClass('text-success');
        }
    }
    
    // Convert markdown to HTML
    function markdownToHTML(markdown) {
        // Simple markdown parser for headers, lists, and paragraphs
        let html = markdown
            // Headers
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            // Lists
            .replace(/^\- (.*$)/gm, '<li>$1</li>')
            // Paragraphs
            .replace(/\n\n/g, '</p><p>')
            // Line breaks
            .replace(/\n/g, '<br>');
        
        // Wrap in paragraph tags if not already
        if (!html.startsWith('<h') && !html.startsWith('<p>')) {
            html = '<p>' + html + '</p>';
        }
        
        return html;
    }
    
    // Handle file upload
    $('#upload-form').submit(function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        // Check if file is selected
        if (!$('#file')[0].files[0]) {
            showAlert('Please select a file to upload.', 'danger');
            return;
        }
        
        updateStatus('data', 'Loading...');
        
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                showAlert(`Successfully loaded ${response.sequence_count} sequences.`);
                updateStatus('data', `Loaded (${response.sequence_count} sequences)`);
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to upload file.', 'danger');
                updateStatus('data', 'Not loaded');
            }
        });
    });
    
    // Handle dummy data generation
    $('#dummy-form').submit(function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        updateStatus('data', 'Generating...');
        
        $.ajax({
            url: '/generate_dummy',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                showAlert(`Successfully generated ${response.sequence_count} dummy sequences.`);
                updateStatus('data', `Loaded (${response.sequence_count} sequences)`);
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to generate dummy data.', 'danger');
                updateStatus('data', 'Not loaded');
            }
        });
    });
    
    // Handle model initialization
    $('#initialize-form').submit(function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        updateStatus('model', 'Initializing...');
        
        $.ajax({
            url: '/initialize_gan',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                showAlert('Successfully initialized GAN model.');
                updateStatus('model', 'Initialized');
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to initialize model.', 'danger');
                updateStatus('model', 'Not initialized');
            }
        });
    });
    
    // Handle model training
    $('#train-form').submit(function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        updateStatus('training', 'In progress...');
        
        $.ajax({
            url: '/train_gan',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                showAlert(`Successfully trained model for ${response.history.epochs} epochs.`);
                updateStatus('training', `Completed (${response.history.epochs} epochs)`);
                
                // Show training history
                $('#training-history-container').show();
                fetchTrainingHistory();
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to train model.', 'danger');
                updateStatus('training', 'Not started');
            }
        });
    });
    
    // Handle checkpoint loading
    $('#checkpoint-form').submit(function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        $.ajax({
            url: '/load_checkpoint',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                showAlert(response.message);
                updateStatus('training', `Loaded (${response.history.epochs} epochs)`);
                
                // Show training history
                $('#training-history-container').show();
                fetchTrainingHistory();
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to load checkpoint.', 'danger');
            }
        });
    });
    
    // Handle sequence generation
    $('#generate-form').submit(function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        updateStatus('generation', 'Generating...');
        
        $.ajax({
            url: '/generate_sequences',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                showAlert(`Successfully generated ${response.sequence_count} sequences.`);
                updateStatus('generation', `Generated (${response.sequence_count} sequences)`);
                
                // Show download button
                $('#download-container').show();
                
                // Preview sequences
                let previewHTML = '<h5 class="mt-4">Generated Sequences Preview</h5>';
                previewHTML += '<div class="sequence-display">';
                
                response.sequences.forEach((seq, i) => {
                    previewHTML += `<div><strong>Sequence ${i+1}:</strong> ${seq}</div>`;
                });
                
                previewHTML += '</div>';
                
                // Append to form
                $(this).append(previewHTML);
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to generate sequences.', 'danger');
                updateStatus('generation', 'Not generated');
            }
        });
    });
    
    // Handle visualization of real sequences
    $('#visualize-real-btn').click(function() {
        showLoading('#visualization-container');
        
        $.ajax({
            url: '/visualize_sequences',
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    // Parse JSON strings to objects
                    const seqViewer = JSON.parse(response.seq_viewer);
                    const kmerDist = JSON.parse(response.kmer_dist);
                    const gcDist = JSON.parse(response.gc_dist);
                    
                    // Create container for visualizations
                    let visualizationHTML = `
                        <div class="row">
                            <div class="col-12">
                                <div class="plot-container">
                                    <div class="plot-title">DNA Sequence Viewer</div>
                                    <div id="seq-viewer-plot" style="width: 100%; height: 300px;"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="plot-container">
                                    <div class="plot-title">K-mer Distribution</div>
                                    <div id="kmer-dist-plot" style="width: 100%; height: 400px;"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="plot-container">
                                    <div class="plot-title">GC Content Distribution</div>
                                    <div id="gc-dist-plot" style="width: 100%; height: 400px;"></div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    $('#visualization-container').html(visualizationHTML);
                    
                    // Render plots
                    Plotly.newPlot('seq-viewer-plot', seqViewer.data, seqViewer.layout);
                    Plotly.newPlot('kmer-dist-plot', kmerDist.data, kmerDist.layout);
                    Plotly.newPlot('gc-dist-plot', gcDist.data, gcDist.layout);
                } else {
                    showAlert(response.error || 'Failed to visualize sequences.', 'danger');
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to visualize sequences.', 'danger');
                $('#visualization-container').html('<div class="alert alert-danger">Failed to load visualizations.</div>');
            }
        });
    });
    
    // Handle visualization comparison
    $('#visualize-comparison-btn').click(function() {
        showLoading('#visualization-container');
        
        $.ajax({
            url: '/visualize_comparison',
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    // Parse JSON strings to objects
                    const realViewer = JSON.parse(response.real_viewer);
                    const synthViewer = JSON.parse(response.synth_viewer);
                    const kmerComp = JSON.parse(response.kmer_comp);
                    const gcComp = JSON.parse(response.gc_comp);
                    
                    // Create container for visualizations
                    let visualizationHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <div class="plot-container">
                                    <div class="plot-title">Real DNA Sequences</div>
                                    <div id="real-viewer-plot" style="width: 100%; height: 250px;"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="plot-container">
                                    <div class="plot-title">Synthetic DNA Sequences</div>
                                    <div id="synth-viewer-plot" style="width: 100%; height: 250px;"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="plot-container">
                                    <div class="plot-title">K-mer Distribution Comparison</div>
                                    <div id="kmer-comp-plot" style="width: 100%; height: 400px;"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="plot-container">
                                    <div class="plot-title">GC Content Comparison</div>
                                    <div id="gc-comp-plot" style="width: 100%; height: 400px;"></div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    $('#visualization-container').html(visualizationHTML);
                    
                    // Render plots
                    Plotly.newPlot('real-viewer-plot', realViewer.data, realViewer.layout);
                    Plotly.newPlot('synth-viewer-plot', synthViewer.data, synthViewer.layout);
                    Plotly.newPlot('kmer-comp-plot', kmerComp.data, kmerComp.layout);
                    Plotly.newPlot('gc-comp-plot', gcComp.data, gcComp.layout);
                } else {
                    showAlert(response.error || 'Failed to visualize comparison.', 'danger');
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to visualize comparison.', 'danger');
                $('#visualization-container').html('<div class="alert alert-danger">Failed to load comparison visualizations.</div>');
            }
        });
    });
    
    // Handle evaluation
    $('#evaluate-btn').click(function() {
        showLoading('#evaluation-container');
        
        $.ajax({
            url: '/evaluate',
            type: 'POST',
            success: function(response) {
                if (response.success) {
                    // Convert markdown report to HTML
                    const reportHTML = markdownToHTML(response.report);
                    
                    // Create container for evaluation results
                    let evaluationHTML = `
                        <div class="markdown-report">
                            ${reportHTML}
                        </div>
                    `;
                    
                    $('#evaluation-container').html(evaluationHTML);
                } else {
                    showAlert(response.error || 'Failed to evaluate sequences.', 'danger');
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to evaluate sequences.', 'danger');
                $('#evaluation-container').html('<div class="alert alert-danger">Failed to evaluate sequences.</div>');
            }
        });
    });
    
    // Handle download
    $('#download-btn').click(function() {
        $.ajax({
            url: '/download_sequences',
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    // Create a temporary link and click it to download
                    const link = document.createElement('a');
                    link.href = response.download_url;
                    link.download = 'synthetic_sequences.csv';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                } else {
                    showAlert(response.error || 'Failed to download sequences.', 'danger');
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON || {};
                showAlert(response.error || 'Failed to download sequences.', 'danger');
            }
        });
    });
    
    // Fetch training history
    function fetchTrainingHistory() {
        $.ajax({
            url: '/training_history',
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    // Parse JSON string to object
                    const historyPlot = JSON.parse(response.history_plot);
                    
                    // Render plot
                    Plotly.newPlot('training-history-plot', historyPlot.data, historyPlot.layout);
                }
            },
            error: function(xhr) {
                console.error('Failed to fetch training history');
            }
        });
    }
    
    // Tab navigation
    $('.list-group-item').on('click', function(e) {
        e.preventDefault();
        
        // Update active tab
        $('.list-group-item').removeClass('active');
        $(this).addClass('active');
        
        // Show corresponding tab content
        const target = $(this).attr('href');
        $('.tab-pane').removeClass('show active');
        $(target).addClass('show active');
    });
});
