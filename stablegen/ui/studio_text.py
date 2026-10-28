"""Small Unicode text buffer with selection and local undo for Studio fields.

Changes are committed to RNA on leaving a field. While typing, Ctrl-Z applies
only to this buffer and never undoes the scene beneath an active UI session.
"""
from dataclasses import dataclass, field


@dataclass
class TextBuffer:
    text: str
    caret: int = 0
    anchor: int = 0
    history: list = field(default_factory=list)
    future: list = field(default_factory=list)

    def __post_init__(self):
        self.caret = self.anchor = len(self.text)

    @property
    def selection(self):
        return min(self.caret, self.anchor), max(self.caret, self.anchor)

    def snapshot(self):
        return self.text, self.caret, self.anchor

    def replace(self, value):
        self.history.append(self.snapshot())
        self.history = self.history[-150:]
        self.future.clear()
        start, end = self.selection
        self.text = self.text[:start] + value + self.text[end:]
        self.caret = self.anchor = start + len(value)

    def move(self, destination, select=False):
        self.caret = max(0, min(len(self.text), destination))
        if not select:
            self.anchor = self.caret

    def word(self, direction):
        index = self.caret
        if direction < 0:
            while index and self.text[index - 1].isspace():
                index -= 1
            while index and not self.text[index - 1].isspace():
                index -= 1
        else:
            while index < len(self.text) and not self.text[index].isspace():
                index += 1
            while index < len(self.text) and self.text[index].isspace():
                index += 1
        return index

    def delete(self, backward=False, word=False):
        if self.caret == self.anchor:
            self.anchor = self.word(-1 if backward else 1) if word else max(0, min(len(self.text), self.caret + (-1 if backward else 1)))
        if self.caret != self.anchor:
            self.replace('')

    def undo(self, redo=False):
        source, target = (self.future, self.history) if redo else (self.history, self.future)
        if source:
            target.append(self.snapshot())
            self.text, self.caret, self.anchor = source.pop()


def wrap(text, width, measure):
    """Return (start, end, displayed text) without changing the source string."""
    result = []
    start = 0
    for paragraph in text.split('\n'):
        if not paragraph:
            result.append((start, start, ''))
        offset = 0
        while offset < len(paragraph):
            end, last_space = offset, None
            while end < len(paragraph):
                if measure(paragraph[offset:end + 1]) > width and end > offset:
                    break
                if paragraph[end].isspace():
                    last_space = end + 1
                end += 1
            if end < len(paragraph) and last_space is not None and last_space > offset:
                end = last_space
            result.append((start + offset, start + end, paragraph[offset:end]))
            offset = end
        start += len(paragraph) + 1
    return result
