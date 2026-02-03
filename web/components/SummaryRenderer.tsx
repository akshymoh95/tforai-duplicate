interface SummaryRendererProps {
  content: string;
}

const HEADINGS = [
  "Overall Summary",
  "Key Findings",
  "Drivers & Diagnostics",
  "Drivers and Diagnostics",
  "Recommendations",
  "Risks",
  "Next Steps",
  "Outlook",
];

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeMarkdown(raw: string) {
  // Backend already provides proper markdown, just clean it up
  let text = (raw || "").replace(/\r\n/g, "\n").trim();
  if (!text) return "";
  
  // Clean up excessive newlines
  text = text.replace(/\n{3,}/g, "\n\n");
  return text;
}

function renderInline(text: string) {
  // Parse markdown: **bold**, *italic*, `code`, **bold:**
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  
  // Match patterns: **text:, **text**, *text*, `code`
  // Use this pattern to catch **word...word**: and **word...word**
  const regex = /\*\*([^*]+)\*\*[:]*|\*([^*]+)\*|`([^`]+)`/g;
  let match;
  
  while ((match = regex.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    
    if (match[1]) {
      // **bold** or **bold**: pattern
      const boldText = match[1];
      const fullMatch = match[0]; // Get the actual matched text including colon if present
      const hasColon = fullMatch.endsWith(":");
      
      parts.push(
        <strong key={`bold-${match.index}`} className="font-semibold text-slate-900">
          {boldText}
        </strong>
      );
      
      // Add colon if it was in the pattern
      if (hasColon) {
        parts.push(":");
      }
    } else if (match[2]) {
      // *italic* pattern
      parts.push(
        <em key={`em-${match.index}`} className="italic text-slate-700">
          {match[2]}
        </em>
      );
    } else if (match[3]) {
      // `code` pattern
      parts.push(
        <code key={`code-${match.index}`} className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded text-xs font-mono">
          {match[3]}
        </code>
      );
    }
    
    lastIndex = regex.lastIndex;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  
  return parts.length > 0 ? parts : [text];
}

function renderBodyContent(text: string) {
  // Split by double newlines to get paragraphs/lists
  const sections = text.split(/\n(?=[-\d*]|\S)/);
  
  return sections.map((section, idx) => {
    const trimmed = section.trim();
    if (!trimmed) return null;
    
    // Check if it's a bullet list
    if (trimmed.match(/^-\s/)) {
      const lines = trimmed.split('\n');
      return (
        <ul key={`ul-${idx}`} className="mt-2 ml-4 space-y-2 list-disc">
          {lines.map((line, i) => {
            const trimmedLine = line.trim();
            if (!trimmedLine) return null;
            
            // Check if this is a bullet item
            if (trimmedLine.match(/^-\s/)) {
              const bulletText = trimmedLine.replace(/^-\s+/, '');
              // Collect all following indented lines as nested content
              const nestedLines = [];
              for (let j = i + 1; j < lines.length; j++) {
                const nextLine = lines[j];
                if (nextLine.match(/^\s+/) && !nextLine.trim().match(/^-\s/)) {
                  nestedLines.push(nextLine.trim());
                } else {
                  break;
                }
              }
              
              return (
                <li key={i} className="text-slate-700">
                  {renderInline(bulletText)}
                  {nestedLines.length > 0 && (
                    <ol className="mt-1 ml-4 space-y-1 list-decimal">
                      {nestedLines.map((nestedLine, ni) => (
                        <li key={ni} className="text-slate-700 text-sm">
                          {renderInline(nestedLine.replace(/^\d+\.\s+/, ''))}
                        </li>
                      ))}
                    </ol>
                  )}
                </li>
              );
            }
            return null;
          }).filter(Boolean)}
        </ul>
      );
    }
    
    // Check if it's a numbered list
    if (trimmed.match(/^\d+\.\s/)) {
      const lines = trimmed.split('\n');
      const numberedItems = lines.filter(l => l.trim().match(/^\d+\.\s/));
      
      if (numberedItems.length > 0) {
        return (
          <ol key={`ol-${idx}`} className="mt-2 ml-4 space-y-1 list-decimal">
            {numberedItems.map((item, i) => (
              <li key={i} className="text-slate-700">
                {renderInline(item.replace(/^\d+\.\s+/, ''))}
              </li>
            ))}
          </ol>
        );
      }
    }
    
    // Regular paragraph
    return (
      <p key={`p-${idx}`} className="mt-2 text-slate-700 leading-relaxed">
        {renderInline(trimmed)}
      </p>
    );
  }).filter(Boolean);
}

export default function SummaryRenderer({ content }: SummaryRendererProps) {
  const normalized = normalizeMarkdown(content);
  const blocks = normalized ? normalized.split(/\n{2,}/).filter(Boolean) : [];

  if (!normalized) {
    return <p className="text-slate-500">Ask a question to generate insights.</p>;
  }

  return (
    <div className="text-slate-700 text-sm leading-relaxed">
      {blocks.map((block, idx) => {
        const trimmed = block.trim();
        const isHeading = /^#{2,3}\s/.test(trimmed);
        const isUnordered = /^-\s/.test(trimmed);
        const isOrdered = /^\d+\.\s/.test(trimmed);

        if (isHeading) {
          // Extract only the first line as the header
          const lines = trimmed.split('\n');
          const headerLine = lines[0];
          const bodyLines = lines.slice(1).join('\n').trim();
          
          const headerText = headerLine.replace(/^#{2,3}\s+/, "").trim();
          
          return (
            <div key={`h-${idx}`} className={idx === 0 ? "" : "mt-6"}>
              {idx !== 0 ? <div className="h-px bg-blue-200/60 mb-3" /> : null}
              <h3 className="text-base font-semibold text-blue-900">{headerText}</h3>
              {bodyLines && (
                <div className="mt-2">
                  {renderBodyContent(bodyLines)}
                </div>
              )}
            </div>
          );
        }

        if (isUnordered) {
          const items = trimmed.split("\n").filter((line) => line.trim().startsWith("- "));
          return (
            <ul key={`ul-${idx}`} className="mt-3 ml-5 space-y-2 list-disc">
              {items.map((line, itemIdx) => (
                <li key={`uli-${itemIdx}`}>{renderInline(line.replace(/^- /, "").trim())}</li>
              ))}
            </ul>
          );
        }

        if (isOrdered) {
          const items = trimmed.split("\n").filter((line) => /^\d+\.\s/.test(line.trim()));
          return (
            <ol key={`ol-${idx}`} className="mt-3 ml-5 space-y-2 list-decimal">
              {items.map((line, itemIdx) => (
                <li key={`oli-${itemIdx}`}>{renderInline(line.replace(/^\d+\.\s+/, "").trim())}</li>
              ))}
            </ol>
          );
        }

        return (
          <p key={`p-${idx}`} className={idx === 0 ? "mt-2" : "mt-3"}>
            {renderInline(trimmed)}
          </p>
        );
      })}
    </div>
  );
}
