document.addEventListener('DOMContentLoaded', () => {
    const taskListEl = document.getElementById('task-list');
    const emptyState = document.getElementById('empty-state');
    const triageFlow = document.getElementById('triage-flow');
    const resetBtn = document.getElementById('reset-btn');
    
    // Phases
    const phaseA = document.getElementById('phase-a');
    const phaseB = document.getElementById('phase-b');
    const phaseC = document.getElementById('phase-c');
    
    // Steps
    const step1 = document.getElementById('step-1');
    const step2 = document.getElementById('step-2');
    const step3 = document.getElementById('step-3');
    const step4 = document.getElementById('step-4');

    // UI Elements for Case Details
    const caseTitleEl = document.getElementById('case-title');
    const caseDiffEl = document.getElementById('case-difficulty');
    const factPatternEl = document.getElementById('fact-pattern');
    const evidenceFlagsEl = document.getElementById('evidence-flags');
    const statutesListEl = document.getElementById('statutes-list');
    const precedentsListEl = document.getElementById('precedents-list');
    
    // AI Elements
    const summonAiBtn = document.getElementById('summon-ai-btn');
    const aiLoading = document.getElementById('ai-loading');
    const aiOutput = document.getElementById('ai-output');
    const resultsOverlay = document.getElementById('results-overlay');
    const decisionMatrix = document.getElementById('decision-matrix');

    // Escalation Modal
    const escalationModal = document.getElementById('escalation-modal');
    
    // State
    let currentDomain = null;
    let currentDifficulty = null;
    let currentCaseData = null;

    // Initialize
    fetchTasks();

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
        
        // Reset Flow
        resetBtn.disabled = true;
        emptyState.style.display = 'none';
        triageFlow.style.display = 'block';
        
        // Reset Phases
        phaseA.style.display = 'block';
        phaseB.style.display = 'none';
        phaseC.style.display = 'none';
        
        // Reset Stepper
        step1.classList.add('active');
        step2.classList.remove('active');
        step3.classList.remove('active');
        step4.classList.remove('active');

        // Reset AI Output
        aiLoading.style.display = 'none';
        aiOutput.style.display = 'none';
        resultsOverlay.style.display = 'none';
        decisionMatrix.style.display = 'block';
        summonAiBtn.style.display = 'block';
        summonAiBtn.disabled = true;
        
        caseTitleEl.textContent = 'Loading Case Data...';
        
        try {
            const res = await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ domain, difficulty })
            });
            const data = await res.json();
            currentCaseData = data.observation;
            caseTitleEl.textContent = `Case Ready for Triage`;
            resetBtn.disabled = false;
        } catch (err) {
            console.error(err);
            caseTitleEl.textContent = 'Error Loading Case';
        }
    }

    // --- Phase A: Triage ---
    document.querySelectorAll('.triage-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const type = e.currentTarget.dataset.type;
            const subCatGrid = document.getElementById('sub-cat-grid');
            subCatGrid.innerHTML = ''; // clear previous

            if (type === 'civil') {
                const cats = ['Family / Matrimonial', 'Property / Real Estate', 'Breach of Contract', 'Tort / Personal Injury'];
                cats.forEach(c => {
                    const b = document.createElement('button');
                    b.className = 'btn secondary-btn';
                    b.textContent = c;
                    b.addEventListener('click', () => proceedToEvidence());
                    subCatGrid.appendChild(b);
                });
            } else {
                const cats = ['Petty Crime', 'Major Felony', 'White Collar'];
                cats.forEach(c => {
                    const b = document.createElement('button');
                    b.className = 'btn secondary-btn';
                    b.textContent = c;
                    b.addEventListener('click', () => proceedToEvidence());
                    subCatGrid.appendChild(b);
                });
            }

            phaseA.style.display = 'none';
            phaseB.style.display = 'block';
            step2.classList.add('active');
        });
    });

    // Back from Phase B
    document.getElementById('back-to-a').addEventListener('click', () => {
        phaseB.style.display = 'none';
        phaseA.style.display = 'block';
        step2.classList.remove('active');
    });

    // --- Phase C: Evidence & Fact Finding ---
    let chatHistoryData = [];
    const chatHistoryEl = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const evidenceDropZone = document.getElementById('evidence-drop-zone');

    function proceedToEvidence() {
        phaseB.style.display = 'none';
        phaseC.style.display = 'grid';
        step3.classList.add('active');
        renderCase(currentCaseData);
        
        // Reset chat
        chatHistoryData = [];
        chatHistoryEl.innerHTML = '';
        summonAiBtn.disabled = true;
        chatInput.value = '';
        
        // Trigger first AI message
        sendChatMessage("");
    }

    async function sendChatMessage(userMessage) {
        if (userMessage) {
            appendChat('user', userMessage);
            chatHistoryData.push({ role: 'user', content: userMessage });
        }

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    case_id: currentCaseData.case_id,
                    fact_pattern: currentCaseData.fact_pattern,
                    user_message: userMessage,
                    chat_history: chatHistoryData
                })
            });
            const data = await res.json();
            appendChat('ai', data.response);
            chatHistoryData.push({ role: 'ai', content: data.response });
            
            // Enable draft resolution if AI is satisfied
            if (chatHistoryData.length >= 4) {
                summonAiBtn.disabled = false;
            }
        } catch(err) {
            console.error(err);
        }
    }

    function appendChat(role, text) {
        const div = document.createElement('div');
        div.style.padding = "0.5rem";
        div.style.borderRadius = "4px";
        div.style.fontSize = "0.9rem";
        if (role === 'user') {
            div.style.background = "rgba(59,130,246,0.3)";
            div.style.marginLeft = "2rem";
            div.innerHTML = `<strong>You:</strong> ${text}`;
        } else {
            div.style.background = "rgba(255,255,255,0.05)";
            div.style.marginRight = "2rem";
            div.style.borderLeft = "2px solid var(--accent-primary)";
            div.innerHTML = `<strong>JusticeEngine-01:</strong> ${text}`;
        }
        chatHistoryEl.appendChild(div);
        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
    }

    chatSendBtn.addEventListener('click', () => {
        const text = chatInput.value.trim();
        if(text) {
            chatInput.value = '';
            sendChatMessage(text);
        }
    });

    chatInput.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') {
            const text = chatInput.value.trim();
            if(text) {
                chatInput.value = '';
                sendChatMessage(text);
            }
        }
    });

    // Evidence Locker drag and drop simulation
    evidenceDropZone.addEventListener('click', () => {
        sendChatMessage("[Uploaded Document: Signed Lease Agreement.pdf]");
        // AI will naturally respond to this based on the mock logic
    });

    function renderCase(obs) {
        caseTitleEl.textContent = `Case ID: ${obs.case_id}`;
        caseDiffEl.textContent = obs.difficulty;
        caseDiffEl.className = `badge ${obs.difficulty}`;
        caseDiffEl.style.display = 'inline-block';

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

        // Precedents
        precedentsListEl.innerHTML = '';
        obs.precedents.forEach((prec) => {
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
        });
    }

    // --- Phase D: AI Resolution ---
    summonAiBtn.addEventListener('click', async () => {
        summonAiBtn.style.display = 'none';
        aiLoading.style.display = 'block';

        try {
            const res = await fetch('/ai_judge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: currentDomain,
                    difficulty: currentDifficulty
                })
            });
            
            if (!res.ok) throw new Error("Server returned " + res.status);
            const data = await res.json();
            
            aiLoading.style.display = 'none';
            aiOutput.style.display = 'block';
            step4.classList.add('active');
            
            // Populate AI Output
            document.getElementById('ai-case-id').textContent = currentCaseData.case_id;
            document.getElementById('ai-verdict').textContent = data.action.verdict;
            document.getElementById('ai-reasoning-text').textContent = data.action.reasoning_chain;
            
            // Save info for later display if accepted
            window.currentEvaluation = data.evaluation.info;
            
        } catch (err) {
            console.error(err);
            summonAiBtn.style.display = 'block';
            aiLoading.style.display = 'none';
            alert('Failed to summon AI Judge. Ensure your API keys are valid.');
        }
    });

    // --- Decision Matrix Actions ---
    document.getElementById('btn-accept').addEventListener('click', () => {
        decisionMatrix.style.display = 'none';
        const info = window.currentEvaluation;
        if(info) {
            document.getElementById('composite-score').textContent = info.composite_reward.toFixed(2);
            document.getElementById('score-logic').textContent = info.logic_score.toFixed(2);
            document.getElementById('score-accuracy').textContent = info.accuracy_score.toFixed(2);
            document.getElementById('score-fairness').textContent = info.fairness_score.toFixed(2);
            document.getElementById('score-citation').textContent = info.citation_score.toFixed(2);
        }
        resultsOverlay.style.display = 'block';
    });

    document.getElementById('btn-refine').addEventListener('click', () => {
        alert("The 'Refine Context' feature is a mock-up for this demo. In a real scenario, this would allow you to upload more evidence.");
    });

    document.getElementById('btn-escalate').addEventListener('click', () => {
        escalationModal.style.display = 'flex';
    });

    // --- Modal Logic ---
    document.getElementById('btn-cancel-transfer').addEventListener('click', () => {
        escalationModal.style.display = 'none';
    });

    document.getElementById('btn-confirm-transfer').addEventListener('click', async () => {
        // Collect checked reasons
        const checks = document.querySelectorAll('#escalation-modal input[type="checkbox"]:checked');
        const reasons = Array.from(checks).map(c => c.parentElement.textContent.trim());
        
        try {
            await fetch('/escalate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    case_id: currentCaseData.case_id,
                    fact_pattern: currentCaseData.fact_pattern,
                    ai_verdict: document.getElementById('ai-verdict').textContent,
                    ai_reasoning: document.getElementById('ai-reasoning-text').textContent,
                    reasons: reasons.length > 0 ? reasons : ["User requested human oversight"]
                })
            });
        } catch (e) {
            console.error("Failed to push to escalate DB", e);
        }

        escalationModal.style.display = 'none';
        decisionMatrix.style.display = 'none';
        resultsOverlay.style.display = 'block';
        resultsOverlay.innerHTML = `
            <h3 style="color:var(--accent-danger)">Case Escalated</h3>
            <p style="color:var(--text-secondary)">All documents and AI preliminary findings have been securely forwarded to a human presiding officer.</p>
        `;
    });

    // Restart
    resetBtn.addEventListener('click', () => {
        if (currentDomain && currentDifficulty) {
            loadCase(currentDomain, currentDifficulty);
        }
    });
});
