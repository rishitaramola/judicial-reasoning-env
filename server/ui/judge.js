document.addEventListener('DOMContentLoaded', async () => {
    const listEl = document.getElementById('escalated-list');
    const countEl = document.getElementById('pending-count');

    try {
        const res = await fetch('/api/escalated-cases');
        const data = await res.json();
        const cases = data.cases;
        
        countEl.textContent = cases.length;

        if (cases.length === 0) {
            listEl.innerHTML = '<div class="text-block" style="text-align:center;">No escalated cases pending. JusticeEngine-01 is handling the load efficiently!</div>';
            return;
        }

        listEl.innerHTML = '';
        cases.reverse().forEach((c, index) => {
            const card = document.createElement('div');
            card.className = 'escalated-card';
            
            // Format reasons
            const reasonsHtml = c.reasons.map(r => `<li>${r}</li>`).join('');

            card.innerHTML = `
                <h3>
                    <span>Case ID: ${c.case_id}</span>
                    <button class="btn primary-btn" style="padding: 0.3rem 0.8rem; font-size:0.85rem;" onclick="alert('Human review workflow not implemented in demo')">Start Review</button>
                </h3>
                
                <div class="escalation-reason">
                    <strong>User Escalation Reasons:</strong>
                    <ul style="margin-top:0.5rem; margin-left:1.5rem;">
                        ${reasonsHtml}
                    </ul>
                </div>

                <div class="grid-2">
                    <div class="case-summary">
                        <strong style="color:var(--accent-primary)">Case Fact Summary</strong>
                        <p class="mt-3 text-block" style="font-size:0.9rem;">${c.fact_pattern}</p>
                    </div>
                    
                    <div class="ai-summary">
                        <strong style="color:#7289da;">JusticeEngine-01 Preliminary Output</strong>
                        <div style="margin-top:1rem;">
                            <strong>Draft Verdict:</strong> <span class="badge" style="background:rgba(255,255,255,0.1)">${c.ai_verdict}</span>
                        </div>
                        <p class="mt-3 text-block" style="font-size:0.9rem; border-left:2px solid #7289da; padding-left:1rem;">
                            ${c.ai_reasoning}
                        </p>
                    </div>
                </div>
            `;
            listEl.appendChild(card);
        });

    } catch (e) {
        listEl.innerHTML = '<div class="text-block" style="color:var(--accent-danger)">Failed to load escalated cases.</div>';
        console.error(e);
    }
});
