// ─── Screen Manager ───────────────────────────────────
function show(id) {
    document.querySelectorAll('.screen').forEach(s => {
        s.classList.remove('active');
        s.style.display = 'none';
    });
    const el = document.getElementById(id);
    el.style.display = 'flex';
    el.classList.add('active');
    window.scrollTo(0, 0);
}

// ─── State ────────────────────────────────────────────
let currentType = null;      // 'civil' | 'criminal'
let currentDomain = null;    // e.g. 'property', 'tort'
let currentDifficulty = null;
let currentCaseData = null;
let chatHistory = [];

// ─── Sub-category data ────────────────────────────────
const CIVIL_CATS = [
    { icon: '🏠', label: 'Property / Rent', desc: 'Disputes about land, home, or rent payments', domain: 'property', difficulty: 'medium' },
    { icon: '📄', label: 'Breach of Contract', desc: 'Someone did not honour a signed agreement', domain: 'contract', difficulty: 'easy' },
    { icon: '👨‍👩‍👧', label: 'Family / Matrimonial', desc: 'Divorce, custody, adoption or maintenance', domain: 'family', difficulty: 'medium' },
    { icon: '⚠️', label: 'Tort / Personal Injury', desc: 'Someone caused harm through negligence', domain: 'tort', difficulty: 'medium' },
    { icon: '💼', label: 'Employment Dispute', desc: 'Wrongful termination, salary or workplace issues', domain: 'contract', difficulty: 'easy' },
    { icon: '🤷', label: 'Other / Not Sure', desc: 'Describe it in your own words to the AI', domain: 'contract', difficulty: 'easy' },
];
const CRIMINAL_CATS = [
    { icon: '🪙', label: 'Petty Crime', desc: 'Minor offences like trespass or public nuisance', domain: 'petty_crime', difficulty: 'easy' },
    { icon: '👊', label: 'Assault / Hurt', desc: 'Physical harm caused by another person', domain: 'petty_crime', difficulty: 'medium' },
    { icon: '💳', label: 'Fraud / Cheating', desc: 'Deception or financial fraud', domain: 'petty_crime', difficulty: 'medium' },
    { icon: '🏠', label: 'Theft / Robbery', desc: 'Unlawful taking of your property', domain: 'petty_crime', difficulty: 'easy' },
    { icon: '📱', label: 'Cyber Crime', desc: 'Online harassment, hacking or identity theft', domain: 'petty_crime', difficulty: 'hard' },
    { icon: '🤷', label: 'Other Offence', desc: 'Describe it — JusticeEngine-01 will categorise', domain: 'petty_crime', difficulty: 'easy' },
];

// ─── Landing ──────────────────────────────────────────
document.getElementById('enter-btn').addEventListener('click', () => show('screen-action'));
document.getElementById('back-to-landing').addEventListener('click', () => show('screen-landing'));

// ─── Action chooser ───────────────────────────────────
document.getElementById('btn-file-case').addEventListener('click', () => show('screen-type'));
document.getElementById('btn-withdraw').addEventListener('click', () => show('screen-withdraw'));
document.getElementById('back-from-withdraw').addEventListener('click', () => show('screen-action'));
document.getElementById('btn-confirm-withdraw').addEventListener('click', () => {
    alert('Case withdrawal request submitted. Reference number sent to your registered contact.');
    show('screen-action');
});

// ─── Civil / Criminal ─────────────────────────────────
document.getElementById('back-to-action').addEventListener('click', () => show('screen-action'));
document.getElementById('btn-civil').addEventListener('click', () => { currentType = 'civil'; buildSubcats(CIVIL_CATS); show('screen-subcat'); });
document.getElementById('btn-criminal').addEventListener('click', () => { currentType = 'criminal'; buildSubcats(CRIMINAL_CATS); show('screen-subcat'); });
document.getElementById('back-to-type').addEventListener('click', () => show('screen-type'));

// ─── Sub-category builder ─────────────────────────────
function buildSubcats(cats) {
    const grid = document.getElementById('subcat-grid');
    grid.innerHTML = '';
    cats.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'subcat-card';
        btn.innerHTML = `
            <span class="sc-icon">${cat.icon}</span>
            <strong>${cat.label}</strong>
            <p>${cat.desc}</p>
        `;
        btn.addEventListener('click', () => {
            currentDomain = cat.domain;
            currentDifficulty = cat.difficulty;
            loadDossier();
        });
        grid.appendChild(btn);
    });
}

// ─── Load Case & Go to Dossier ────────────────────────
async function loadDossier() {
    show('screen-dossier');
    document.getElementById('dossier-badge').textContent = currentType === 'civil' ? 'Civil Case' : 'Criminal Case';

    // Reset right panel
    document.getElementById('chat-panel').style.display = 'block';
    document.getElementById('ai-thinking').style.display = 'none';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('accepted-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'none';
    document.getElementById('generate-btn').disabled = true;
    document.getElementById('generate-hint').textContent = 'Answer the questions above to unlock the judgment';
    document.getElementById('chat-messages').innerHTML = '';
    chatHistory = [];

    // Fetch case from backend
    try {
        const res = await fetch('/reset', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ domain: currentDomain, difficulty: currentDifficulty })
        });
        const data = await res.json();
        currentCaseData = data.observation;
        renderDossierLeft(currentCaseData);
        startFactFinding();
    } catch(e) {
        renderDossierLeft({ case_id: 'DEMO-001', fact_pattern: 'Could not load case from server.', evidence_flags: [], statutes: [] });
        startFactFinding();
    }
}

document.getElementById('back-to-subcat').addEventListener('click', () => show('screen-subcat'));

// ─── Render left dossier ──────────────────────────────
function renderDossierLeft(obs) {
    document.getElementById('d-fact-pattern').textContent = obs.fact_pattern;

    const evEl = document.getElementById('d-evidence');
    evEl.innerHTML = '';
    (obs.evidence_flags || []).forEach(f => {
        const s = document.createElement('span');
        s.textContent = f;
        evEl.appendChild(s);
    });
    if (!obs.evidence_flags || obs.evidence_flags.length === 0) evEl.innerHTML = '<span style="color:var(--muted);font-size:0.85rem">None provided yet</span>';

    const stEl = document.getElementById('d-statutes');
    stEl.innerHTML = '';
    (obs.statutes || []).forEach(s => {
        const li = document.createElement('li');
        li.textContent = s;
        stEl.appendChild(li);
    });
}

// ─── Fact Finding Chat ────────────────────────────────
function startFactFinding() {
    postAI("To help me build your legal dossier, I need to ask you a few short questions. Let's begin — could you confirm whether you have any written proof related to your case, such as a contract, receipt, or message?");
}

async function sendUserMessage(text) {
    if (!text.trim()) return;
    appendMsg('user', text);
    chatHistory.push({ role: 'user', content: text });
    document.getElementById('chat-input').value = '';

    try {
        const res = await fetch('/chat', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                case_id: currentCaseData ? currentCaseData.case_id : 'DEMO',
                fact_pattern: currentCaseData ? currentCaseData.fact_pattern : '',
                user_message: text,
                chat_history: chatHistory
            })
        });
        const data = await res.json();
        postAI(data.response);
        chatHistory.push({ role: 'ai', content: data.response });
    } catch(e) {
        postAI("I have gathered enough information. You may proceed to generate the AI judgment.");
    }

    // Unlock judgment after 4 messages
    if (chatHistory.length >= 4) {
        document.getElementById('generate-btn').disabled = false;
        document.getElementById('generate-hint').textContent = '✅ Dossier ready — you may now generate the judgment';
    }
}

function postAI(text) {
    const div = document.createElement('div');
    div.className = 'msg msg-ai';
    div.innerHTML = `<strong>JusticeEngine-01</strong>${text}`;
    document.getElementById('chat-messages').appendChild(div);
    document.getElementById('chat-messages').scrollTop = 9999;
}

function appendMsg(role, text) {
    const div = document.createElement('div');
    div.className = role === 'user' ? 'msg msg-user' : 'msg msg-ai';
    div.textContent = text;
    document.getElementById('chat-messages').appendChild(div);
    document.getElementById('chat-messages').scrollTop = 9999;
}

document.getElementById('chat-send').addEventListener('click', () => sendUserMessage(document.getElementById('chat-input').value));
document.getElementById('chat-input').addEventListener('keypress', e => { if(e.key === 'Enter') sendUserMessage(document.getElementById('chat-input').value); });

// Evidence locker click
document.getElementById('locker-zone').addEventListener('click', () => {
    sendUserMessage('[Document uploaded: Evidence file submitted to dossier]');
});

// ─── Generate Judgment ────────────────────────────────
document.getElementById('generate-btn').addEventListener('click', async () => {
    document.getElementById('chat-panel').style.display = 'none';
    document.getElementById('ai-thinking').style.display = 'block';

    try {
        const res = await fetch('/ai_judge', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ domain: currentDomain, difficulty: currentDifficulty })
        });
        if (!res.ok) throw new Error();
        const data = await res.json();

        document.getElementById('ai-thinking').style.display = 'none';
        document.getElementById('verdict-panel').style.display = 'block';

        document.getElementById('v-case-id').textContent = currentCaseData ? currentCaseData.case_id : 'N/A';
        document.getElementById('v-verdict').textContent = data.action.verdict;
        document.getElementById('v-reasoning').textContent = data.action.reasoning_chain;
        window.__evalInfo = data.evaluation.info;
    } catch(e) {
        document.getElementById('ai-thinking').style.display = 'none';
        document.getElementById('chat-panel').style.display = 'block';
        document.getElementById('generate-btn').disabled = false;
        postAI("⚠️ I was unable to generate a judgment at this time. Please ensure the server API key is configured and try again.");
    }
});

// ─── Accept ───────────────────────────────────────────
document.getElementById('btn-accept').addEventListener('click', () => {
    document.getElementById('verdict-panel').style.display = 'none';
    const panel = document.getElementById('accepted-panel');
    panel.style.display = 'block';

    const info = window.__evalInfo;
    const row = document.getElementById('metrics-row');
    row.innerHTML = '';
    if (info) {
        const items = [
            { label: 'Logic', val: info.logic_score },
            { label: 'Accuracy', val: info.accuracy_score },
            { label: 'Fairness', val: info.fairness_score },
            { label: 'Citation', val: info.citation_score },
        ];
        items.forEach(i => {
            const chip = document.createElement('div');
            chip.className = 'metric-chip';
            chip.innerHTML = `${i.label}: <span>${(i.val||0).toFixed(2)}</span>`;
            row.appendChild(chip);
        });
    }
});

// ─── Escalate ─────────────────────────────────────────
document.getElementById('btn-escalate').addEventListener('click', () => {
    document.getElementById('escalation-modal').style.display = 'flex';
});
document.getElementById('modal-cancel').addEventListener('click', () => {
    document.getElementById('escalation-modal').style.display = 'none';
});
document.getElementById('modal-confirm').addEventListener('click', async () => {
    const reasons = ['r1','r2','r3','r4']
        .filter(id => document.getElementById(id).checked)
        .map(id => document.getElementById(id).parentElement.textContent.trim());

    try {
        await fetch('/escalate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                case_id: currentCaseData ? currentCaseData.case_id : 'N/A',
                fact_pattern: currentCaseData ? currentCaseData.fact_pattern : '',
                ai_verdict: document.getElementById('v-verdict').textContent,
                ai_reasoning: document.getElementById('v-reasoning').textContent,
                reasons: reasons.length ? reasons : ['User requested human oversight']
            })
        });
    } catch(e) { console.error(e); }

    document.getElementById('escalation-modal').style.display = 'none';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'block';
});

// ─── Init ─────────────────────────────────────────────
show('screen-landing');
