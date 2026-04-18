import './globals.css';

export const metadata = {
  title: 'AI Driven Legal Support & Justice Access System',
  description: 'Artificial Intelligence Driven Legal Support and Information System to Enhance Access to Justice for the General Public.',
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
