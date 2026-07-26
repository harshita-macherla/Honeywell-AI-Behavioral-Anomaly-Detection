import { useCallback, useEffect, useRef, useState } from "react";

/**
 * useApi
 * ======
 * Small fetch-lifecycle hook: runs `fetcher` whenever `deps` change, and
 * exposes { data, loading, error, reload }. Guards against setting state
 * after unmount and against stale responses overwriting a newer request
 * (relevant here since filters/pagination can change quickly on the
 * Alerts Queue page).
 */
export function useApi(fetcher, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const requestIdRef = useRef(0);
  const mountedRef = useRef(true);

  const load = useCallback(() => {
    const thisRequestId = ++requestIdRef.current;
    setLoading(true);
    setError(null);

    fetcher()
      .then((result) => {
        if (!mountedRef.current || thisRequestId !== requestIdRef.current) return;
        setData(result);
        setLoading(false);
      })
      .catch((err) => {
        if (!mountedRef.current || thisRequestId !== requestIdRef.current) return;
        setError(err);
        setLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mountedRef.current = true;
    load();
    return () => {
      mountedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  return { data, loading, error, reload: load };
}
