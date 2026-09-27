"""Yanıt metnini motor kanıtından kurar, modelin yazdığından değil."""

import json

GROUNDING_CHECK = (
    "LLM selects retrieved source IDs only. Quoted text is copied from those sources; "
    "transaction facts are rendered from engine evidence. Relevance and source accuracy "
    "still require evaluation; this is not a free-form semantic truth verifier."
)


def render_answer(selection, sources, evidence=None):
    """Keep policy quotations visibly separate from the current transaction decision."""
    tr = selection.language == "tr"
    lines = []
    limitations = [
        "Skor fraud olasılığı değildir; karar inceleme önerisidir, güvenlik onayı değildir."
        if tr else "The score is not a fraud probability; the decision is an investigation recommendation, not a safety approval.",
        "Kaynak seçiminin soruyla ilgisi ve bilgi tabanının doğruluğu ayrıca değerlendirilmelidir."
        if tr else "Source relevance and knowledge-base accuracy require separate evaluation.",
    ]
    if evidence is not None:
        decision, winner = evidence["decision"], evidence.get("winning_rule")
        labels = {"review": "inceleme önerisi", "monitor": "izleme önerisi",
                  "no_rule_match": "eşleşen kural yok; güvenli olduğu anlamına gelmez"}
        lines.append((f"Motor kararı: {decision} ({labels[decision]}). Kazanan kural: {winner or 'yok'}."
                      if tr else f"Engine decision: {decision}. Winning rule: {winner or 'none'}.") )
        matched = evidence.get("matched_rules", [])
        winning = next((r for r in matched if r["id"] == winner), None)
        if winning:
            observed = json.dumps(winning["observations"], ensure_ascii=False, allow_nan=False, sort_keys=True)
            lines.append((f"Kazanan kuralın önceliği {winning['priority']}; gözlenen değerler: {observed}."
                          if tr else f"Winning rule priority: {winning['priority']}; observed values: {observed}."))
            if "condition" in winning:
                lines.append(("Koşul ve eşik kanıtı: " if tr else "Condition and threshold evidence: ") +
                             json.dumps(winning["condition"], ensure_ascii=False, allow_nan=False, sort_keys=True))
        others = [r["id"] for r in matched if r["id"] != winner]
        if others:
            lines.append(("Diğer eşleşmeler: " if tr else "Other matched rules: ") + ", ".join(others) + ". " +
                         ("Yalnız eşleşen kurallar arasında yüksek öncelik, eşitlikte alfabetik ID kazanır."
                          if tr else "Only matched rules compete: highest priority wins, then ascending ID breaks ties."))
        if not winner:
            lines.append("Eşleşme olmaması işlemi onaylamaz." if tr else "No matching rule does not approve the transaction.")
        values = {key: evidence[key] for key in ("raw_anomaly_score", "adjusted_anomaly_score", "context_profile") if key in evidence}
        if values:
            lines.append(("Motor skorları/profili: " if tr else "Engine scores/profile: ") +
                         json.dumps(values, ensure_ascii=False, allow_nan=False, sort_keys=True) + ".")
        limitations.extend(evidence.get("limitations", []))
    if selection.abstained:
        lines.append("Soruyu yanıtlayan yeterli kaynak seçilemedi." if tr else "No sufficient source was selected to answer the question.")
    quotes = []
    by_id = {source["id"]: source for source in sources}
    for source_id in selection.source_ids:
        source = by_id[source_id]
        quotes.append({"source_id": source_id, "text": source["text"]})
        # Kaynağı olduğu gibi al - model olumsuzluğu, kaybeden bütçeyi veya
        # geçerlilik koşulunu kırpmasın
        lines.append((f"Kaynak metni [{source_id}] (genel politika, işlem kararı değil):\n"
                      if tr else f"Source text [{source_id}] (general policy, not the transaction decision):\n") + source["text"])
    lines.append(("Sınır: " if tr else "Limitation: ") + limitations[0])
    return {"answer": "\n\n".join(lines), "citations": list(selection.source_ids),
            "limitations": list(dict.fromkeys(limitations)), "abstained": selection.abstained,
            "answer_mode": "source_selection_with_engine_facts", "source_quotes": quotes,
            "authoritative_evidence": evidence, "grounding_check": GROUNDING_CHECK}
