import csv
import json
import os

from openai import OpenAI

SYSTEM_MESSAGE = (
    "List every line of direct speech in the passage (the words inside 「」), in order. "
    "For each line give: speaker = the name of the character who says it; "
    "when the narrator speaks as 我, give the narrator's name if the passage lets you infer it; "
    "mention = the word the passage uses for the speaker just before the line "
    "(for example 我, 她, or a name); text = the words inside 「」, without the brackets. "
    "Sounds and actions such as 嗯了一聲 are not speech. "
    "Use only the passage. Keep everything in Chinese."
)

EXAMPLE_PASSAGE = (
    "她那時滿臉是汗，直起腰來還埋怨我：「你幹你的，過來幹什麼？」她是怕我這麼一過去，"
    "別人都注意她了，我說：「你自己留意著身體。」她急了，說：「你快走開。」我搖搖頭，"
    "只好走開。……家珍哭了，她說：「福貴，我還能養活自己嗎？」「能。」我說。"
)

EXAMPLE_JSON = {
    "utterances": [
        {"speaker": "家珍", "mention": "她", "text": "你幹你的，過來幹什麼？"},
        {"speaker": "福貴", "mention": "我", "text": "你自己留意著身體。"},
        {"speaker": "家珍", "mention": "她", "text": "你快走開。"},
        {"speaker": "家珍", "mention": "她", "text": "福貴，我還能養活自己嗎？"},
        {"speaker": "福貴", "mention": "我", "text": "能。"},
    ]
}

JSON_SCHEMA = {
    "name": "dialogue",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "utterances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "mention": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "mention", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["utterances"],
        "additionalProperties": False,
    },
}


def main():
    with open("paragraph.txt", encoding="utf-8") as f:
        passage = f.read().strip()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.5-9b",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": EXAMPLE_PASSAGE},
            {"role": "assistant", "content": json.dumps(EXAMPLE_JSON, ensure_ascii=False)},
            {"role": "user", "content": passage},
        ],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        extra_body={"reasoning": {"enabled": False}},
    )

    data = json.loads(response.choices[0].message.content)

    print(json.dumps(data, ensure_ascii=False, indent=2))

    with open("dialogue.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open("dialogue.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "speaker", "mention", "text"])
        writer.writeheader()
        for i, u in enumerate(data["utterances"], start=1):
            writer.writerow(
                {
                    "id": i,
                    "speaker": u["speaker"],
                    "mention": u["mention"],
                    "text": u["text"],
                }
            )


if __name__ == "__main__":
    main()
