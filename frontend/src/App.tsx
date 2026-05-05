import { Routes, Route, NavLink } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ClaimsPage from './pages/ClaimsPage';
import ClaimDetailPage from './pages/ClaimDetailPage';
import RemittancePage from './pages/RemittancePage';
import DashboardPage from './pages/DashboardPage';

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/upload', label: 'Upload' },
  { to: '/claims', label: 'Claims' },
  { to: '/remittances', label: 'Remittances' },
];

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <span className="text-xl font-bold text-indigo-600">RCM AI Demo</span>
              <div className="ml-10 flex space-x-4">
                {navItems.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === '/'}
                    className={({ isActive }) =>
                      `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        isActive
                          ? 'bg-indigo-100 text-indigo-700'
                          : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ))}
              </div>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/claims" element={<ClaimsPage />} />
          <Route path="/claims/:claimId" element={<ClaimDetailPage />} />
          <Route path="/remittances" element={<RemittancePage />} />
        </Routes>
      </main>
    </div>
  );
}
