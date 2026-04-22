document.addEventListener('DOMContentLoaded', () => {
    const taskListEl = document.getElementById('task-list');
    const mainGrid = document.getElementById('main-grid');
    const emptyState = document.getElementById('empty-state');
    const resetBtn = document.getElementById('reset-btn');
    const submitBtn = document.getElementById('submit-btn');
    const actionForm = document.getElementById('action-form');
    
    // UI Elements for Case Details
    const caseTitleEl = document.getElementById('case-title');
    const caseDiffEl = document.getElementById('case-difficulty');
    const factPatternEl = document.getElementById('fact-pattern');
    const evidenceFlagsEl = document.getElementById('evidence-flags');
    const statutesListEl = document.getElementById('statutes-list');
    const precedentsListEl = document.getElementById('precedents-list');
    const citationOptionsEl = document.getElementById('citation-options');
    const resultsOverlay = document.getElementById('results-overlay');
    
    // State
    let currentDomain = null;
    let currentDifficulty = null;
    let currentCaseId = null;

    // Initialize
    fetchTasks();

    // Confidence Slider Sync
    const confidenceSlider = document.getElementById('confidence');
    const confidenceVal = document.getElementById('confidence-val');
    confidenceSlider.addEventListener('input', (e) => {
        confidenceVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    async function fetchTasks() {
        try {
            const res = await fetch('/tasks');
            const data = await res.json();
            renderTaskList(data.tasks);
        } catch (err) {
            taskListEl.innerHTML = '<div class="loading-spinner" style="color:var(--accent-danger)">Failed to load tasks. Ensure server is running.</div>';
            console.error(err);
        }
    }

    function renderTaskList(tasks) {
        taskListEl.innerHTML = '';
        tasks.forEach(task => {
            const card = document.createElement('div');
            card.className = 'task-card';
            card.innerHTML = `
                <div class="task-name">${task.id}</div>
                <div class="task-domain">
                    <span>${task.domain}</span>
                    <span class="badge ${task.difficulty}">${task.difficulty}</span>
                </div>
            `;
            card.addEventListener('click', () => {
                document.querySelectorAll('.task-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                loadCase(task.domain, task.difficulty);
            });
            taskListEl.appendChild(card);
        });
    }

    async function loadCase(domain, difficulty) {
        currentDomain = domain;
        currentDifficulty = difficulty;
        
        // UI Reset
        resetBtn.disabled = true;
        emptyState.style.display = 'none';
        mainGrid.style.display = 'grid';
        resultsOverlay.style.display = 'none';
        actionForm.reset();
        confidenceVal.textContent = '0.85';
        confidenceSlider.value = 0.85;
        submitBtn.disabled = false;
        
        caseTitleEl.textContent = 'Loading Case...';
        factPatternEl.textContent = 'Loading...';

        try {
            const res = await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ domain, difficulty })
            });
            const data = await res.json();
            renderCase(data.observation);
            resetBtn.disabled = false;
        } catch (err) {
            console.error(err);
            caseTitleEl.textContent = 'Error Loading Case';
        }
    }

    function renderCase(obs) {
        currentCaseId = obs.case_id;
        caseTitleEl.textContent = `Case: ${obs.case_id}`;
        caseDiffEl.textContent = obs.difficulty;
        caseDiffEl.className = `badge ${obs.difficulty}`;

        factPatternEl.textContent = obs.fact_pattern;

        // Evidence Flags
        evidenceFlagsEl.innerHTML = '';
        if (obs.evidence_flags && obs.evidence_flags.length > 0) {
            obs.evidence_flags.forEach(flag => {
                const span = document.createElement('span');
                span.className = 'tag';
                span.textContent = flag;
                evidenceFlagsEl.appendChild(span);
            });
        } else {
            evidenceFlagsEl.innerHTML = '<span style="color:var(--text-secondary); font-size:0.85rem">None</span>';
        }

        // Statutes
        statutesListEl.innerHTML = '';
        obs.statutes.forEach(statute => {
            const li = document.createElement('li');
            li.textContent = statute;
            statutesListEl.appendChild(li);
        });

        // Precedents & Citation Options
        precedentsListEl.innerHTML = '';
        citationOptionsEl.innerHTML = '';
        
        obs.precedents.forEach((prec, index) => {
            // Accordion for viewing
            const item = document.createElement('div');
            item.className = 'accordion-item';
            item.innerHTML = `
                <div class="accordion-title">${prec.case_id}: ${prec.title || 'Precedent'}</div>
                <div class="accordion-content">
                    <strong>Verdict:</strong> ${prec.verdict}<br>
                    <strong>Rationale:</strong> ${prec.rationale}
                </div>
            `;
            precedentsListEl.appendChild(item);

            // Checkbox for citing
            const label = document.createElement('label');
            label.className = 'checkbox-label';
            label.innerHTML = `
                <input type="checkbox" name="cited_precedents" value="${prec.case_id}">
                ${prec.case_id}
            `;
            citationOptionsEl.appendChild(label);
        });
        
        // Add a "hallucination" trap option for fun (optional, to test RL agent behavior)
        const trapLabel = document.createElement('label');
        trapLabel.className = 'checkbox-label';
        trapLabel.innerHTML = `
            <input type="checkbox" name="cited_precedents" value="FAKE_999">
            FAKE_999 (Hallucinated Precedent)
        `;
        citationOptionsEl.appendChild(trapLabel);
    }

    actionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const verdict = document.getElementById('verdict').value;
        const confidence = parseFloat(document.getElementById('confidence').value);
        const reasoning = document.getElementById('reasoning').value;
        
        const citedElements = document.querySelectorAll('input[name="cited_precedents"]:checked');
        const cited = Array.from(citedElements).map(el => el.value);

        const action = {
            verdict: verdict,
            confidence_score: confidence,
            reasoning_chain: reasoning,
            cited_precedents: cited
        };

        submitBtn.disabled = true;
        submitBtn.textContent = 'Evaluating...';

        try {
            const res = await fetch('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: currentDomain,
                    difficulty: currentDifficulty,
                    action: action
                })
            });
            const data = await res.json();
            showResults(data.info);
        } catch (err) {
            console.error(err);
            submitBtn.disabled = false;
            submitBtn.textContent = 'Submit Verdict';
            alert('Failed to evaluate verdict.');
        }
    });

    function showResults(info) {
        document.getElementById('composite-score').textContent = info.composite_reward.toFixed(2);
        document.getElementById('score-logic').textContent = info.logic_score.toFixed(2);
        document.getElementById('score-accuracy').textContent = info.accuracy_score.toFixed(2);
        document.getElementById('score-fairness').textContent = info.fairness_score.toFixed(2);
        document.getElementById('score-citation').textContent = info.citation_score.toFixed(2);

        resultsOverlay.style.display = 'block';
        submitBtn.style.display = 'none'; // hide submit
    }

    resetBtn.addEventListener('click', () => {
        if (currentDomain && currentDifficulty) {
            loadCase(currentDomain, currentDifficulty);
            submitBtn.style.display = 'block';
            submitBtn.textContent = 'Submit Verdict';
        }
    });

    document.getElementById('next-case-btn').addEventListener('click', () => {
        if (currentDomain && currentDifficulty) {
            loadCase(currentDomain, currentDifficulty);
            submitBtn.style.display = 'block';
            submitBtn.textContent = 'Submit Verdict';
        }
    });
});
