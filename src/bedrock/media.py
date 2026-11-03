class MediaBudget:
    IMAGE_STUB = "[Earlier image dropped to stay within the per-request image limit — re-run the tool that produced it to see it again.]"
    DOCUMENT_STUB = "[Earlier document dropped to stay within the per-request document limit — re-run the tool that produced it to see it again.]"
    VIDEO_STUB = "[Earlier video dropped to stay within the per-request media limit — re-run the tool that produced it to watch it again.]"

    def __init__(self, max_images=20, keep_images=12, max_documents=5, keep_documents=3, max_media_bytes=18_000_000, keep_media_bytes=12_000_000):
        self.max_images = max_images
        self.keep_images = keep_images
        self.max_documents = max_documents
        self.keep_documents = keep_documents
        self.max_media_bytes = max_media_bytes
        self.keep_media_bytes = keep_media_bytes

    def apply(self, messages):
        kept = []
        for pair in list(self.media_pairs(messages)):
            kept = self.settled([*kept, pair])
        return messages

    def settled(self, kept):
        self.trim([pair for pair in kept if pair[1].image], self.max_images, self.keep_images)
        self.trim([pair for pair in kept if pair[1].document], self.max_documents, self.keep_documents)
        self.trim_bytes([pair for pair in kept if self.is_media(pair[1])])
        return [pair for pair in kept if self.is_media(pair[1])]

    def is_media(self, part):
        return bool(part.image or part.document or part.video)

    def media_pairs(self, messages):
        for message in messages:
            for part in message.content:
                if self.is_media(part):
                    yield message, part
                if part.tool_result:
                    yield from ((message, item) for item in part.tool_result.content if self.is_media(item))

    def trim(self, pairs, cap, keep):
        for message, part in pairs[:len(pairs) - keep] if len(pairs) > cap else []:
            self.stub(message, part)

    def trim_bytes(self, pairs):
        total = sum(self.part_bytes(part) for _, part in pairs)
        for message, part in pairs if total > self.max_media_bytes else []:
            if total <= self.keep_media_bytes:
                return
            total -= self.part_bytes(part)
            self.stub(message, part)

    def part_bytes(self, part):
        media = part.image or part.document or part.video
        return len(media.source.bytes)

    def stub(self, message, part):
        part.text = self.IMAGE_STUB if part.image else self.VIDEO_STUB if part.video else self.document_text(part.document)
        part.image, part.document, part.video = None, None, None
        message.content = list(message.content)

    def document_text(self, document):
        return self.DOCUMENT_STUB
