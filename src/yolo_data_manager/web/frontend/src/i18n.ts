import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import enTranslation from './locales/en.json';
import zhTranslation from './locales/zh.json';

export const localeMap: Record<string, string> = {
  en: 'en-US',
  zh: 'zh-CN',
};

function getInitialLanguage(): 'en' | 'zh' {
  const saved = window.localStorage.getItem('ydm.language');
  if (saved === 'en' || saved === 'zh') return saved;
  return window.navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
}

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: enTranslation },
      zh: { translation: zhTranslation },
    },
    lng: getInitialLanguage(),
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
  });

i18n.on('languageChanged', (language) => {
  const normalized = language.startsWith('zh') ? 'zh' : 'en';
  window.localStorage.setItem('ydm.language', normalized);
  document.documentElement.lang = localeMap[normalized];
});

document.documentElement.lang = localeMap[i18n.language.startsWith('zh') ? 'zh' : 'en'];

export default i18n;
