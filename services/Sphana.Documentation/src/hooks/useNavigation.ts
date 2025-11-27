import { useState, useEffect } from 'react';
import type { Navigation } from '../types/navigation';

export function useNavigation() {
  const [navigation, setNavigation] = useState<Navigation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    fetch('/content/navigation.json')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load navigation');
        return res.json();
      })
      .then(data => {
        setNavigation(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err);
        setLoading(false);
      });
  }, []);

  return { navigation, loading, error };
}

