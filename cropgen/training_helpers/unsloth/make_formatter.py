def prepare_unsloth_formatter(instruction_text: str):
    def formatter(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction_text,
                        },
                        {
                            "type": "image",
                            "image": sample["image"],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": sample["text"],
                        }
                    ],
                },
            ]
        }

    return formatter
