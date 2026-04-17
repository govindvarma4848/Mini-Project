import './globals.css';

export const metadata = {
  title: 'LexSumm | Legal Document Summarizer',
  description: 'AI-powered legal document summarization and retrieval system.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
